import json
import time
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import DashboardModelSettings, DashboardVideoSettings, Appearance
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
import folium
import threading
import pytz
import torch
from datetime import datetime
import numpy as np
from . import loggers
from . import camera

# mxnet and gluoncv must be built from source
# install CPU version of mxnet and gluoncv before running this
from . import detection

from picode import pi_publisher
from picode import pi_subscriber

import os
import cv2
from django.conf import settings
pacific_tz = pytz.timezone('US/Pacific')
model_weights = os.path.join(
    settings.BASE_DIR, "model_weights/patrolNanoWeights.pt")
model_weight_path = model_weights
yolo = torch.hub.load("ultralytics/yolov5", "custom", model_weight_path)
classes = yolo.names
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
runActionDetection = False

# robot movement
commandTime = int(time.time())
pendingCommand = False
commandDelay = 4
movementDirection = 'none'

# robot gps
longitude = 48.864716
latitude = 2.349014

# model settings
model_settings = {"Bounding Box Overlay (object detection model)": True, "Person": True, "Bike": True,
                  "Bolt Cutters": True, "Angle Grinder": True}


def welcome_view(request):
    return render(request, "pages/welcome.html", {})


def security_logs(request):
    pacific_tz = pytz.timezone("US/Pacific")
    time_of_request = datetime.now(pacific_tz).strftime("%Y-%M-%d")
    data_for_request = {
        "time_of_request": time_of_request,
        "logs": loggers.security_notices,
        "whichlogs": "threat_log",
    }
    return JsonResponse(data_for_request)


def action_logs(request):
    pacific_tz = pytz.timezone("US/Pacific")
    time_of_request = datetime.now(pacific_tz).strftime("%Y-%M-%d")
    data_for_request = {
        "time_of_request": time_of_request,
        "logs": loggers.objects_detected,
        "whichlogs": "action_log",
    }
    return JsonResponse(data_for_request)


def refresh_map_view(request):
    # [latitude, longitude]

    global latitude
    global longitude

    subscriber = pi_subscriber.Subscriber()
    subscriber.subscribe("robot/location", gps_callback)
    print("lat read in refresh_map_view")
    print(latitude)
    print("lon read in refresh_map_view")
    print(longitude)
    new_robot_location = [latitude, longitude]
    f = folium.Figure(width="100%", height="100%")
    m = folium.Map(
        location=new_robot_location,
        zoom_start=20,
        dragging=False,
        scrollWheelZoom=False,
        attributionControl=False,
        zoom_control=False,
    ).add_to(f)
    folium.Marker(new_robot_location).add_to(m)
    m = m._repr_html_()
    return JsonResponse({"map": m})


@csrf_exempt
def get_direction_data(request):
    global commandTime
    global pendingCommand
    global commandDelay
    global movementDirection
    if request.method == "POST":
        direction = request.POST["direction"]
        if direction == "N" or direction == "NE" or direction == "NW":
            if pendingCommand == False:
                commandTime = int(time.time())
                # print("forward")
                pendingCommand = True
                movementDirection = "f"
        elif direction == "S" or direction == "SE" or direction == "SW":
            if pendingCommand == False:
                # print("backward")
                commandTime = int(time.time())
                pendingCommand = True
                movementDirection = "b"
        elif direction == "W":
            if pendingCommand == False:
                # print("left")
                commandTime = int(time.time())
                pendingCommand = True
                movementDirection = "l"
        elif direction == "E":
            if pendingCommand == False:
                # print("right")
                commandTime = int(time.time())
                pendingCommand = True
                movementDirection = "r"

        if int(time.time()) >= commandTime + commandDelay and pendingCommand == True:
            print("execution time")
            commandTime = int(time.time())
            pendingCommand = False

            if(movementDirection == 'f'):
                # print(movementDirection)
                pi_publisher.forward()
            elif(movementDirection == 'b'):
                # print(movementDirection)
                pi_publisher.backward()
            elif(movementDirection == 'l'):
                # print(movementDirection)
                pi_publisher.turn_left()
            elif(movementDirection == 'r'):
                # print(movementDirection)
                pi_publisher.turn_right()

        # handle sending commands here
        return render(request, "pages/dashboard.html", {})


def gps_callback(self, params, packet):
    global longitude
    global latitude
    payload = json.loads(packet.payload)
    lat = payload["lat"]
    lon = payload["lon"]
    longitude = lon
    latitude = lat
    print("gps callback issued")
    print("payload lat:")
    print(lat)
    print("payload lon")
    print(lon)
    print("read lat")
    print(latitude)
    print("read lon")
    print(longitude)


@login_required
def dashboard_view(request):
    if request.user.is_authenticated:
        robo_coords = [longitude, latitude]

        f = folium.Figure(width="100%", height="100%")
        # create map object
        m = folium.Map(
            location=robo_coords,
            zoom_start=20,
            dragging=False,
            scrollWheelZoom=False,
            attributionControl=False,
            zoom_control=False,
        ).add_to(f)
        folium.Marker(robo_coords).add_to(m)

        # get html representation of map object
        m = m._repr_html_()

        # On a new application version, AWS clears all data on the instance (including database)
        # Add setting records to db only once
        try:
            bounding_box = DashboardVideoSettings.objects.get(
                name_id="Bounding Box Overlay (object detection model)"
            )
        except DashboardVideoSettings.DoesNotExist:
            obj = DashboardVideoSettings(
                name_id="Bounding Box Overlay (object detection model)",
                setting="bounding_box_overlay",
                switch=True,
            )
            obj.save()

        try:
            person_detect = DashboardModelSettings.objects.get(
                name_id="Person")
            bike_detect = DashboardModelSettings.objects.get(name_id="Bike")
            angle_grinder_detect = DashboardModelSettings.objects.get(
                name_id="Angle Grinders"
            )
            bolt_cutter_detect = DashboardModelSettings.objects.get(
                name_id="Bolt Cutters"
            )
        except DashboardModelSettings.DoesNotExist:
            new_name_ids = ["Person", "Bike", "Angle Grinders", "Bolt Cutters"]
            new_setting_names = [
                "detect_people",
                "detect_bike",
                "detect_angle_grinders",
                "detect_bolt_cutters",
            ]
            for i in range(len(new_name_ids)):
                obj = DashboardModelSettings(
                    name_id=new_name_ids[i], setting=new_setting_names[i], switch=True
                )
                obj.save()
                
        try:
            theme = Appearance.objects.get(appearance="theme")
        except Appearance.DoesNotExist:
            obj = Appearance(appearance="theme", theme=True)
            obj.save()

        # Pass in context for rendered template
        context = {
            "m": m,
        }
        return render(request, "pages/dashboard.html", context)
    else:
        return redirect("/")


@login_required
def dashboard_settings_view(request):
    detection.toggle_detection()
    global model_settings    
    if request.user.is_authenticated:
        # Get video settings from database
        video_settings = DashboardVideoSettings.objects.all()

        # Get model settings from database
        model_setting = DashboardModelSettings.objects.all()

        if request.method == "POST":
            # Update settings accordingly...

            # Uncheck all settings
            DashboardModelSettings.objects.all().order_by("id").update(switch=False)
            DashboardVideoSettings.objects.all().order_by("id").update(switch=False)

            # Set all global settings to false
            for key in model_settings:
                model_settings[key] = False

            # Get list of checkbed box id's
            video_settings_id_list = request.POST.getlist(
                "dashboard_video_settings")
            model_settings_id_list = request.POST.getlist(
                "dashboard_model_settings")

            # Update the database
            for i in video_settings_id_list:
                DashboardVideoSettings.objects.filter(
                    name_id=i).update(switch=True)
                model_settings[i] = True
            for i in model_settings_id_list:
                DashboardModelSettings.objects.filter(
                    name_id=i).update(switch=True)
                model_settings[i] = True

            messages.success(request, "Settings updated successfully!")
            return render(
                request,
                "pages/settings.html",
                {"video_settings": video_settings,
                    "model_settings": model_setting},
            )
            
        theme = Appearance.objects.get(appearance="theme")
        return render(
            request,
            "pages/settings.html",
            {"video_settings": video_settings, "model_settings": model_setting, "theme": theme.theme},
        )
    else:
        return redirect("/")


# def dashboard_robot_manual_view(request):
#    if request.method == "POST":
#        if "forward_command" in request.POST:
#            pi_publisher.forward()
#        elif "backward_command" in request.POST:
#            pi_publisher.backward()
#        elif "turn_left_command" in request.POST:
#            pi_publisher.turn_left()
#        elif "turn_right_command" in request.POST:
#            pi_publisher.turn_right()
#    return render(request, "pages/robot_manual.html", {})


def gen(url):
    global model_settings
    vcap = cv2.VideoCapture(url)
    while True:

        # Extract the names of the classes for trained the YoloV5 model
        class_ids = [0, 1, 2, 3]

        # Capture frame-by-frame
        ret, frame = vcap.read()

        if ret:

            # Flip video frame, so it isn't reversed
            image = cv2.flip(frame, 1)

            # If model is turned on and the object is initialized
            # run object detection on each frame

            ################################################################
            # TORCH OBJECT DETECTION
            ################################################################

            # Get dimensions of the current video frame
            x_shape = image.shape[1]
            y_shape = image.shape[0]

            # Apply the Torch YoloV5 model to this frame
            results = yolo(image)

            # Extract the labels and coordinates of the bounding boxes
            labels, cords = (
                results.xyxyn[0][:, -1].numpy(),
                results.xyxyn[0][:, :-1].numpy(),
            )

            numberOfLabels = len(labels)

            # declare empty array of objects found
            objectsFound = []

            for i in range(numberOfLabels):
                # If global enable flag is set true then show boxes
                if model_settings["Bounding Box Overlay (object detection model)"] == True:
                    row = cords[i]
                    # Get the class number of current label
                    class_number = int(labels[i])
                    # Index colors list with current label number
                    color = COLORS[class_ids[class_number]]

                    # If confidence level is greater than 0.2
                    if row[4] >= 0.4:
                        # Get label to send to dashbaord
                        label = classes[class_number]
                        x1, y1, x2, y2 = (
                            int(row[0] * x_shape),
                            int(row[1] * y_shape),
                            int(row[2] * x_shape),
                            int(row[3] * y_shape),
                        )

                        # if malicious item detected, send alert
                        if label == "Angle Grinder" or label == "Bolt Cutters":
                            # send an alert to the alerts log
                            time_of_event = datetime.now(
                                pacific_tz).strftime("%Y-%m-%d %H:%M:%S")
                            seconds = int(datetime.today().timestamp() % 10)
                            if seconds == 0:
                                # every 10 seconds append to the log form
                                loggers.objects_detected.append(
                                    [time_of_event, "Malicious item detected: " + label])

                        # append coords and label so it can be analyzed
                        objectsFound.append([x1, y1, x2, y2, label])

                        # send objected detected to log page
                        time_of_event = datetime.now(
                            pacific_tz).strftime("%Y-%m-%d %H:%M:%S")
                        seconds = int(datetime.today().timestamp() % 10)
                        if seconds == 0:
                            # every 10 seconds append to the log form
                            loggers.objects_detected.append(
                                [time_of_event, "Object detected: " + label])
                        if model_settings[label] == True:
                            # Draw bounding box
                            cv2.rectangle(image, (int(x1), int(y1)),
                                          (int(x2), int(y2)), color, 2)
                            # Give bounding box a text label
                            cv2.putText(
                                image,
                                str(classes[int(labels[i])]),
                                (int(x1) - 10, int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                color,
                                2,
                            )

            # ensure there are enough objects for action detection algorithm
            if len(objectsFound) >= 2:
                # iterate over items found
                for index in range(len(objectsFound)):
                    # check if first object is malicious
                    if objectsFound[index][4] == 'Angle Grinder' or objectsFound[index][4] == 'Bolt Cutters':

                        # check if it is bike was detected in same frame
                        for index2 in range(len(objectsFound)):
                            if objectsFound[index2][4] == 'Bike':
                                x1, y1, x2, y2, label1 = objectsFound[index]
                                x3, y3, x4, y4, label2 = objectsFound[index2]
                                box1 = [x1, y1, x2, y2]
                                box2 = [x3, y3, x4, y4]
                                # if interseciton is greater than 50 percent
                                # send a threat alert to alert logs
                                iou = iou(box1, box2)
                                if iou >= 0.05:
                                    time_of_event = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    seconds = int(
                                        datetime.today().timestamp() % 10)
                                    if seconds == 0:
                                        # every 10 seconds append to the log form

                                        loggers.security_notices.append(
                                            [time_of_event, "Potential Threat computed with confidence level of " + iou + ": " + label1 + " and " + label2 + " detected"])

            # return the resulting image
            _, jpeg = cv2.imencode(".jpg", image)
            frame = jpeg.tobytes()
            # return frame
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


def phone_feed_view(request):
    return StreamingHttpResponse(
        gen(camera.IPPhoneCamera()),
        content_type="multipart/x-mixed-replace;boundary=frame",
    )

# code from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    def iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou


def kinesis_stream_view(request):
    # retrieves url on hls stream
    url = detection.hls_stream()
    # create action detection model
    action = detection.get_action_model()
    # create thread to run action detection
    detection.toggle_detection()
    thread = threading.Thread(
        target=detection.run_action_detection, args=(url, action, request))
    thread.start()
    return StreamingHttpResponse(
        gen(url),
        content_type="multipart/x-mixed-replace;boundary=frame",
    )


def model_meta_data(request):
    # Secret page that renders model meta data
    is_cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if is_cuda_available else "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return render(
        request,
        "pages/model_meta_data.html",
        {
            "is_cuda_available": is_cuda_available,
            "device_name": device_name,
            "device": device,
        },
    )
