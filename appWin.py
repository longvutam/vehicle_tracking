import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from gui import *
import copy
import sys
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import imutils
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class App(QMainWindow,Ui_mainWindow):
    def __init__(self):
        super(App,self).__init__()
        self.setupUi(self)
        self.label_image_size = (self.label_image.geometry().width(),self.label_image.geometry().height())
        self.video = None
        self.exampleImage = None
        self.imgScale = None
        self.get_points_flag = 0
        self.countArea = []
        self.road_code = None
        self.time_code = None
        self.show_label = ['car','motorbike','bus','bicycle']

        #button function
        self.pushButton_selectArea.clicked.connect(self.select_area)
        self.pushButton_openVideo.clicked.connect(self.open_video)
        self.pushButton_start.clicked.connect(self.start_count)
        self.pushButton_pause.clicked.connect(self.pause)
        self.label_image.mouseDoubleClickEvent = self.get_points


        self.pushButton_selectArea.setEnabled(False)
        self.pushButton_start.setEnabled(False)
        self.pushButton_pause.setEnabled(False)

        #some flags
        self.running_flag = 0
        self.pause_flag = 0
        self.counter_thread_start_flag = 1

        self.framework = 'tf'
        self.weights = './checkpoints/yolov4-416'
        self.size = 416
        self.tiny = False
        self.model = 'yolov4'
        self.output = None
        self.output_format = 'XVID'
        self.iou = 0.45
        self.score = 0.50
        self.dont_show = False
        self.info = False
        self.count = False
        self.CountArea = [[15, 200], [695, 200], [695, 400], [15, 400]]
        self.history = {}
    # Definition of the parameters
        self.max_cosine_distance = 0.4
        self.nn_budget = None
        self.nms_max_overlap = 1.0
    
    
    # initialize deep sort
        self.model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
    # calculate cosine distance metric
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
    # initialize tracker
        self.tracker = Tracker(self.metric)

    # load configuration for object detector
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.input_size = self.size
    # video_path = FLAGS.video

    # load tflite model if flag is set
        if self.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
    # otherwise load standard tensorflow saved model
        else:
            self.saved_model_loaded = tf.saved_model.load(self.weights, tags=[tag_constants.SERVING])
            self.infer = self.saved_model_loaded.signatures['serving_default']


    def open_video(self):
        openfile_name = QFileDialog.getOpenFileName(self,'Open video','','Video files(*.avi , *.mp4)')
        self.videoList = [openfile_name[0]]

        # opendir_name = QFileDialog.getExistingDirectory(self, "Open dir", "./")
        # self.videoList = [os.path.join(opendir_name,item) for item in os.listdir(opendir_name)]
        # self.videoList = list(filter(lambda x: not os.path.isdir(x) , self.videoList))
        # self.videoList.sort()

        vid = cv2.VideoCapture(self.videoList[0])

        # self.videoWriter = cv2.VideoWriter(openfile_name[0].split("/")[-1], cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, (1920, 1080))

        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)
                self.imgScale = np.array(frame.shape[:2]) / [self.label_image_size[1], self.label_image_size[0]]
                vid.release()
                break

        self.pushButton_selectArea.setEnabled(True)
        self.pushButton_start.setText("Start")
        self.pushButton_start.setEnabled(False)
        self.pushButton_pause.setText("Pause")
        self.pushButton_pause.setEnabled(False)

        #clear counting results
        self.label_sum.setText("0")
        self.label_sum.repaint()


    def get_points(self, event):
        if self.get_points_flag:
            x = event.x()
            y = event.y()
            self.countArea.append([int(x*self.imgScale[1]),int(y*self.imgScale[0])])
            exampleImageWithArea = copy.deepcopy(self.exampleImage)
            for point in self.countArea:
                exampleImageWithArea[point[1]-10:point[1]+10,point[0]-10:point[0]+10] = (0,255,255)
            cv2.fillConvexPoly(exampleImageWithArea, np.array(self.countArea), (0,0,255))
            self.show_image_label(exampleImageWithArea)
        print(self.countArea)


    def select_area(self):

        #change Area needs update exampleImage
        if self.counter_thread_start_flag:
            ret, frame = cv2.VideoCapture(self.videoList[0]).read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)

        if not self.get_points_flag:
            self.pushButton_selectArea.setText("Submit Area")
            self.get_points_flag = 1
            self.countArea = []
            self.pushButton_openVideo.setEnabled(False)
            self.pushButton_start.setEnabled(False)

        else:
            self.pushButton_selectArea.setText("Select Area")
            self.get_points_flag = 0
            exampleImage = copy.deepcopy(self.exampleImage)
            # painting area
            for i in range(len(self.countArea)):
                cv2.line(exampleImage, tuple(self.countArea[i]), tuple(self.countArea[(i + 1) % (len(self.countArea))]), (0, 0, 255), 2)
            self.show_image_label(exampleImage)

            #enable start button
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_start.setEnabled(True)


    def show_image_label(self, img_np):
        img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, self.label_image_size)
        frame = QImage(img_np, self.label_image_size[0], self.label_image_size[1], QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.label_image.setPixmap(pix)
        self.label_image.repaint()

    def start_count(self):
        if self.running_flag == 0:
            #clear count and display
           
            for item in self.show_label:
                vars(self)[f"label_{item}"].setText('0')
            # clear result file
            
            frame_num = 0
            #start
            self.running_flag = 1
            self.pause_flag = 0
            self.pushButton_start.setText("Stop")
            self.pushButton_openVideo.setEnabled(False)
            self.pushButton_selectArea.setEnabled(False)
            #emit new parameter to counter thread
            vid = cv2.VideoCapture(self.videoList[0])
            out = None
            while True:
                return_value, frame = vid.read()
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                else:
                    print('Video has ended or failed, try a different video format!')
                    break
                frame_num +=1
                print('Frame #: ', frame_num)
                frame_size = frame.shape[:2]
                image_data = cv2.resize(frame, (self.input_size, self.input_size))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                start_time = time.time()
                laser_line_color = (0, 0, 255)
        # run detections on tflite if flag is set
                if self.framework == 'tflite':
                    interpreter.set_tensor(input_details[0]['index'], image_data)
                    interpreter.invoke()
                    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
                    if self.model == 'yolov3' and tiny == True:
                        boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([self.input_size, self.input_size]))
                    else:
                        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([self.input_size, self.input_size]))
                else:
                    batch_data = tf.constant(image_data)
                    pred_bbox = self.infer(batch_data)
                    for key, value in pred_bbox.items():
                        boxes = value[:, :, 0:4]
                        pred_conf = value[:, :, 4:]

                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=self.iou,
                    score_threshold=self.score
                )

        # convert data to numpy arrays and slice out unused elements
                num_objects = valid_detections.numpy()[0]
                bboxes = boxes.numpy()[0]
                bboxes = bboxes[0:int(num_objects)]
                scores = scores.numpy()[0]
                scores = scores[0:int(num_objects)]
                classes = classes.numpy()[0]
                classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
                original_h, original_w, _ = frame.shape
                bboxes = utils.format_boxes(bboxes, original_h, original_w)

                # store all predictions in one parameter for simplicity when calling functions
                pred_bbox = [bboxes, scores, classes, num_objects]

                # read in all class names from config
                class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        # chỉnh đối tượng được chọn để nhận diện
        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
                allowed_classes = ['car','motorbike','bus','bicycle']

                # loop through objects and use class index to get class name, allow only classes in allowed_classes list
                names = []
                deleted_indx = []
                for i in range(num_objects):
                    class_indx = int(classes[i])
                    class_name = class_names[class_indx]
                    if class_name not in allowed_classes:
                        deleted_indx.append(i)
                    else:
                        names.append(class_name)
                names = np.array(names)
                count = len(names)
                if count:
                    cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                    print("Objects being tracked: {}".format(count))
                # delete detections that are not in allowed_classes
                bboxes = np.delete(bboxes, deleted_indx, axis=0)
                scores = np.delete(scores, deleted_indx, axis=0)

                # encode yolo detections and feed to tracker
                features = self.encoder(frame, bboxes)
                detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

                #initialize color map
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.class_name for d in detections])
                indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
                detections = [detections[i] for i in indices]       

                # Call the tracker
                self.tracker.predict()
                self.tracker.update(detections)

                # update tracks
                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    id = track.track_id - 1
                    if id not in self.history.keys():  #add new id
                        self.history[id] = {}
                        self.history[id]["no_update_count"] = 0
                        self.history[id]["his"] = []
                        self.history[id]["his"].append(class_name)
                    else:
                        self.history[id]["no_update_count"] = 0
                        self.history[id]["his"].append(class_name)
                    print(self.history)

                
            
        # draw bbox on screen
                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                # if enable info flag then print details about each track
                    if self.info:
                        print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                
                counter_results = []

                removed_id_list = []
                        # print(history)
                for id in self.history.keys():    #extract id after tracking
                    self.history[id]["no_update_count"] += 1
                    # nếu 1 object  xuất hiện 5 frame liên tiếp thì sẽ được cộng 1
                    if  self.history[id]["no_update_count"] > 5:
                        his = self.history[id]["his"]
                        result = {}
                        for i in set(his):
                            result[i] = his.count(i)
                        res = sorted(result.items(), key=lambda d: d[1], reverse=True)
                        objectName = res[0][0]
                        #kết quả đếm của từng xe.
                        counter_results.append([id,objectName])
                        
                        removed_id_list.append(id) 
                for id in removed_id_list:
                    _ = self.history.pop(id)
                for i, result in enumerate(counter_results):
                    label_var = vars(self)[f"label_{result[2]}"]
                    label_var.setText(str(int(label_var.text())+1))
                    label_var.repaint()
                    label_sum_var = vars(self)[f"label_sum"]
                    label_sum_var.setText(str(int(label_sum_var.text()) + 1))
                    label_sum_var.repaint()
            
                # calculate frames per second of running detections
                fps = 1.0 / (time.time() - start_time)
                print("FPS: %.2f" % fps)
                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # if not FLAGS.dont_show:
                    # cv2.imshow("Output Video", result)
                self.show_image_label(result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                cv2.destroyAllWindows()
                self.pushButton_pause.setEnabled(True)


        elif self.running_flag == 1:  #push pause button
            #stop system
            self.running_flag = 0
            
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_selectArea.setEnabled(True)
            self.pushButton_start.setText("Start")



    def done(self,sin):
        if sin == 1:
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_start.setEnabled(False)
            self.pushButton_start.setText("Start")


    

    def pause(self):
        if self.pause_flag == 0:
            self.pause_flag = 1
            self.pushButton_pause.setText("Continue")
            self.pushButton_start.setEnabled(False)
        else:
            self.pause_flag = 0
            self.pushButton_pause.setText("Pause")
            self.pushButton_start.setEnabled(True)

        self.counterThread.sin_pauseFlag.emit(self.pause_flag)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = App()
    myWin.show()
    sys.exit(app.exec_())
