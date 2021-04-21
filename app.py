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

from flask import Flask, request, Response, jsonify, send_from_directory, abort,render_template

appp = Flask(__name__)

tt = {
        'car': 0,
        'motorbike': 0,
        'bus': 0,
        'bicycle': 0,  
    }

@appp.route('/')
def home():
    return render_template('index.html')


def gen():
    framework = 'tf'
    weights = './checkpoints/yolov4-416'
    size = 416
    tiny = False
    model = 'yolov4'
    video = 'videoblocks-2.mp4'
    output = None
    output_format = 'XVID'
    iou = 0.45
    score = 0.50
    dont_show = False
    info = False
    count = False
    CountArea = [[15, 200], [695, 200], [695, 400], [15, 400]]
    history = {}
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    history_id = []
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = size
    # video_path = FLAGS.video

    # load tflite model if flag is set
    if framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    laser_line = 990
    # begin video capture
    vid = cv2.VideoCapture(video)
    out = None
    
    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        
            
            
            frame_num +=1
            print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()
            laser_line_color = (0, 0, 255)
            # run detections on tflite if flag is set
            if framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if model == 'yolov3' and tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=score
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
                #cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                print("Objects being tracked: {}".format(count))
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                id = track.track_id - 1
                if id not in history.keys():  #add new id
                    history[id] = {}
                    history[id]["no_update_count"] = 0
                    history[id]["his"] = []
                    history[id]["his"].append(class_name)
                else:
                    history[id]["no_update_count"] = 0
                    history[id]["his"].append(class_name)
                print(history)

            
                
            # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            # if enable info flag then print details about each track
                if info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            cv2.line(frame, (0, laser_line), (1100, laser_line), laser_line_color, 2)
            counter_results = []

            removed_id_list = []
                    # print(history)
            for id in history.keys():    #extract id after tracking
                history[id]["no_update_count"] += 1
                # nếu 1 object  xuất hiện 5 frame liên tiếp thì sẽ được cộng 1
                if  history[id]["no_update_count"] > 5:
                    his = history[id]["his"]
                    result = {}
                    for i in set(his):
                        result[i] = his.count(i)
                    res = sorted(result.items(), key=lambda d: d[1], reverse=True)
                    objectName = res[0][0]
                    #kết quả đếm của từng xe.
                    counter_results.append([id,objectName])
                    if(objectName == "motorbike"):
                        tt['motorbike'] += 1
                                # with open("static/results1.txt", "w") as f:
                                #     f.write(str(tt['motorbike']))
                    elif(objectName == "car"):
                        tt['car'] += 1
                                # with open("static/results2.txt", "w") as f:
                                #     f.write(str(tt['car']))
                    elif(objectName == 'bus'):
                        tt['bus'] += 1
                                # with open("static/results3.txt", "w") as f:
                                #     f.write(str(tt['bus']))
                    else:
                        tt['bicycle'] += 1
                                # with open("static/results4.txt", "w") as f:
                                #     f.write(str(tt['bicycle']))
                    print(tt)
                        
                            # car = str(tt['car'])
                            # motorbike=str(tt['motorbike'])
                            # bus=str(tt['bus']) 
                            # bicycle= str(tt['bicycle'])
                            #del id
                            
                    removed_id_list.append(id) 
            for id in removed_id_list:
                _ = history.pop(id)
            
            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # if not FLAGS.dont_show:
                # cv2.imshow("Output Video", result)
            result = cv2.imencode('.jpg', result)[1].tobytes()
                
            yield(b'--result\r\n'b'Content-Type: image/jpeg\r\n\r\n' + result +b'\r\n')
            
            # if output flag is set, save video file
            # if FLAGS.output:
            #     out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        else:
            vid = cv2.VideoCapture(video)
            tt['car'] = 0
            tt['motorbike'] = 0
            tt['bus'] = 0
            tt['bicycle'] = 0
            frame_num = 0

    cv2.destroyAllWindows()

@appp.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=result')

@appp.route("/stream")
def stream():
    def generate():    
        yield "{}\n".format(tt['car'])
        yield "{}\n".format(tt['motorbike'])
        yield "{}\n".format(tt['bus'])
        yield "{}\n".format(tt['bicycle'])

    return appp.response_class(generate(), mimetype="text/plain")

if __name__ == '__main__':
    appp.run(debug=True, host = '0.0.0.0',port='9999')