function loadDoc() {
    var xhttp1 = new XMLHttpRequest();
    xhttp1.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
        var li = this.responseText.split("\n");
        var sum = parseFloat(li[0]) + parseFloat(li[1]) + parseFloat(li[2]) + parseFloat(li[3]);
        document.getElementById('car').innerHTML = li[0];
        document.getElementById('car-mobile-p').innerHTML = li[0];
        document.getElementById('car-mobile').style.width = ""+(parseFloat(li[0])/sum)*100 + "%";
        document.getElementById('motorbike').innerHTML = li[1];
        document.getElementById('motorbike-mobile-p').innerHTML = li[1];
        document.getElementById('motorbike-mobile').style.width = ""+parseFloat(li[1])/sum*100 + "%";
        document.getElementById('bus').innerHTML = li[2];
        document.getElementById('bus-mobile-p').innerHTML = li[2];
        document.getElementById('bus-mobile').style.width = ""+parseFloat(li[2])/sum*100 + "%";
        document.getElementById('bicycle').innerHTML = li[3];
        document.getElementById('bicycle-mobile-p').innerHTML = li[3];
        document.getElementById('bicycle-mobile').style.width = ""+parseFloat(li[3])/sum*100 + "%";
        document.getElementById('sum').innerHTML = sum;
      }
    };
    xhttp1.open("GET", "http://127.0.0.1:5000/stream", true);
    xhttp1.send();
  }

  function realtimeClock(){
    var rtclock = new Date();
    var hour = rtclock.getHours();
    var minute = rtclock.getMinutes();
    var second = rtclock.getSeconds();
  
    var ampm = (hour < 12 ) ? "AM" : "PM";
    hour = (hour > 12) ? hour - 12 : hour;
  
    // hour = ("0" + hour).slice(-2);
    // minute = ("0" + minute).slice(-2);
    // second = ("0" + second).slice(-2);
  
    document.getElementById('clock').innerHTML = hour + "  :  " + minute + "  :  " + second + " " + ampm;
    
  }

setInterval(loadDoc, 5000);
setInterval(realtimeClock, 500);

// function autocomplete(){
//   var countries = []
//   for(let i = 0 ; i< 237; i++)
//   {
//       countries.push(arrgAll[i*399]['country']);
//   }
//   var searchInput = document.getElementById("searchTable1");
//   var suggestionClass = document.getElementById("suggestion")

//   searchInput.addEventListener('keyup',function(e){
//   const input = searchInput.value ;
//   suggestionClass.innerHTML = '';
//   const suggestions = countries.filter(function(c){
//       return c.toLowerCase().startsWith(input);
//   }); 
//   suggestions.forEach(function(suggested){
//       const div = document.createElement('div');
//       div.innerHTML = '<span class="text-black-50 pl-3"><i class="fas fa-search mr-2"></i></span>'+suggested;
//       suggestionClass.appendChild(div);
//       div.addEventListener('click',function(){
//           searchInput.value = suggested;
//           show(searchInput.value);
//           suggestionClass.innerHTML = '';
//       });
//   });
//   if(input === '')
//   {
//       suggestionClass.innerHTML = '';
//   }
// });
// }