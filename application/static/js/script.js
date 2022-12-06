var map;
var start_point = [];
var linePartArr = [];
var city = 'abakan';

var markers = [];
var infoWindows = [];
var polylines = [];
var asyncs = [];

var compareMarkers = []
var compareInfoWindows = []
var comparePolylines = []
var compareAsync = []

function clearOverlays(objectArray) {
  for (let i = 0; i < objectArray.length; i++ ) {
    objectArray[i].setMap(null);
  }
  objectArray.length = 0;
}

function initMap(callback) {

    map = new google.maps.Map(document.getElementById("map"), {
        zoom: 13,
        center: {lat: 53.7149157, lng: 91.4401772},
        disableDefaultUI: true,
        styles: [
            {
                "featureType": "administrative",
                "elementType": "geometry",
                "stylers": [
                    {
                        "visibility": "off"
                    }
                ]
            },
            {
                "featureType": "poi",
                "stylers": [
                    {
                        "visibility": "off"
                    }
                ]
            },
            {
                "featureType": "road",
                "elementType": "labels.icon",
                "stylers": [
                    {
                        "visibility": "off"
                    }
                ]
            },
            {
                "featureType": "transit",
                "stylers": [
                    {
                        "visibility": "off"
                    }
                ]
            }
        ]
    });

    // Omsk navigation button
    $(document).on('click', '.omsk', function (e) {
        map.setCenter({lat: 54.9867855, lng: 73.3698403})
        city = 'omsk'
    })

    // Abakan navigation button
    $(document).on('click', '.abakan', function (e) {
        map.setCenter({lat: 53.7149157, lng: 91.4401772})
        city = 'abakan'
    })

    // Clear all routes
    $(document).on('click', '.clear', function (e) {
        clearOverlays(markers)
        clearOverlays(infoWindows)
        for (let i = 0; i < asyncs.length; i++) {
            clearInterval(asyncs[i]);
        }
        clearOverlays(linePartArr)
    })

    // Compare baselines on existing routes
    $(document).on('click', '.compare', function (e) {
        let data = new FormData()
        data.append('city', city)

        fetch('/predefined_data/', {
            "method": "POST",
            "body": data,
        })
            .then(function (response) {
                return response.json();
            }).then(function (text) {
            console.log(text);
            let predefined_routes = text['predefined_routes']
            for (let i = 0; i < predefined_routes.length; i++) {
                let color = 'rgb(36,36,36)'

                animatePolyline(predefined_routes[i]['coords'], color, 1);

                // Start marker
                let begin_marker = new google.maps.Marker({
                    position: predefined_routes[i]['coords'][0],
                    map: map,
                });
                markers.push(begin_marker)
                // End marker
                let end_marker = new google.maps.Marker({
                    position: predefined_routes[i]['coords'][predefined_routes[i]['coords'].length -1],
                    map: map,
                });
                markers.push(end_marker)

                // Window with the predicted ETA
                const infoWindow = new google.maps.InfoWindow({
                    content:
                        'MURAT: ' + Math.floor(predefined_routes[i]['preds']['murat']).toString() + ' s<br>' +
                        'WDR: ' + Math.floor(predefined_routes[i]['preds']['wdr']).toString() + ' s<br>' +
                        'DeepI2T: ' + Math.floor(predefined_routes[i]['preds']['deepit2']).toString() + ' s<br>' +
                        'GCT-TTE: ' + Math.floor(predefined_routes[i]['preds']['mtte']).toString() + ' s<br> <hr>' +
                        '<b>Ground truth ETA:</b> ' + predefined_routes[i]['preds']['rta'].toString() + ' s<br>',
                    position: predefined_routes[i]['coords'][predefined_routes[i]['coords'].length -1],
                    maxWidth: 200
                });
                infoWindow.open(map);
                infoWindows.push(infoWindow)

            }
        });
    })

    map.addListener("click", (mapsMouseEvent) => {
        console.log(mapsMouseEvent.latLng.toJSON())

        if (start_point.length === 1) {
            start_point.push(mapsMouseEvent.latLng.toJSON());

            let data = new FormData()
            data.append('start_point', [start_point[0]['lat'], start_point[0]['lng']])
            data.append('end_point', [start_point[1]['lat'], start_point[1]['lng']])
            data.append('city', city)

            start_point = []

            fetch('/getdata/', {
                "method": "POST",
                "body": data
            }).then(function (response) {
                return response.json();
            }).then(function (text) {
                console.log(text);

                let r = Math.floor((Math.random() - 0.5) * 80) // randomized color of lines
                let color = 'rgb(' + (Math.min(20 + r, 0)).toString() + ',' + (135 + r).toString() + ',' + (163 + r).toString() + ')'

                // Create animated polyline
                animatePolyline(text['coords'], color, 0);

                // Start marker
                let begin_marker = new google.maps.Marker({
                    position: text['coords'][0],
                    map: map,
                });
                markers.push(begin_marker)

                // End marker
                let end_marker = new google.maps.Marker({
                    position: text['coords'][text['coords'].length - 1],
                    map: map,
                });
                markers.push(end_marker)

                // Window with the predicted ETA
                const infoWindow = new google.maps.InfoWindow({
                    content: '<b>GCT-TTE:</b> ' + text['tte'].toString() + ' s',
                    position: text['coords'][text['coords'].length - 1],
                    maxWidth: 200
                });
                infoWindow.open(map);
                infoWindows.push(infoWindow)
            });
        } else {
            start_point.push(mapsMouseEvent.latLng.toJSON());
        }
    });
}



function animatePolyline(lineCoordinates, color, status) {
    var i = 0;
    var drawSpeed = 5;
    var pause = false;
    let draw = setInterval(function () {
        if (i + 1 == lineCoordinates.length && !pause) {
            pause = true;
            setTimeout(function () {
                pause = false;
                i = 0;
            }, 100);
        }
        if (!pause) {
            var part = [];
            part.push(lineCoordinates[i]);
            part.push(lineCoordinates[i + 1]);

            var linePart = new google.maps.Polyline({
                path: part,
                strokeColor: color,
                strokeOpacity: 1,
                strokeWeight: 3.5,
                zIndex: i + 2,
                map: map
            });
            linePartArr.push(linePart);
            i++;
        }
    }, drawSpeed);
    asyncs.push(draw)
}

window.initMap = initMap;