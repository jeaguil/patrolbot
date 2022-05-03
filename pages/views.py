""" 
Aggregate implementations for:

Camera Feed
Object Detection Model
Threat Algorithm
Email
Recordings
Settings
GPS
Logs
Robot Controls 
"""

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files import File
from django.db import IntegrityError

from .models import (
    DashboardModelSettings,
    DashboardVideoSettings,
    Appearance,
    Recordings,
    EmailPreferences,
)
from . import loggers
from . import detection

# pub and sub for website
from picode import pi_publisher, pi_subscriber

import os
import cv2
import json
import time
import folium
import threading
import pytz
import torch
from datetime import datetime
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

pacific_tz = pytz.timezone("US/Pacific")
model_weights = os.path.join(settings.BASE_DIR, "model_weights/best.pt")
model_weight_path = model_weights
yolo = torch.hub.load("ultralytics/yolov5", "custom", model_weight_path)
classes = yolo.names
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# robot gps default (university campus)
longitude = 39.539822
latitude = -119.811992

# model settings
model_settings = {
    "Bounding Box Overlay": True,
    "Person": True,
    "Bike": True,
    "Bolt Cutters": True,
    "Angle Grinder": True,
    "Object Detection": True,
}


def welcome_view(request):
    # Render template for welcome url.
    return render(request, "pages/welcome.html", {})


@login_required
def dashboard_view(request):
    # Render dashboard home template for matching url.

    if request.user.is_authenticated:
        robo_coords = [longitude, latitude]

        f = folium.Figure(width="100%", height="100%")
        # create map object
        m = folium.Map(
            location=robo_coords,
            zoom_start=20,
            dragging=True,
            scrollWheelZoom=True,
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
                name_id="Bounding Box Overlay"
            )
            obj_detection = DashboardVideoSettings.objects.get(
                name_id="Object Detection"
            )
        except DashboardVideoSettings.DoesNotExist:
            new_name_id = ["Bounding Box Overlay", "Object Detection"]
            new_setting_names = ["bounding_box_overlay", "object_detection"]
            for i in range(len(new_name_id)):
                obj = DashboardVideoSettings(
                    name_id=new_name_id[i], setting=new_setting_names[i], switch=True
                )
                obj.save()

        try:
            person_detect = DashboardModelSettings.objects.get(name_id="Person")
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
            theme = Appearance.objects.get(appearance="theme")

        try:
            email_preference = EmailPreferences.objects.get(id=1)
        except EmailPreferences.DoesNotExist:
            try:
                obj = EmailPreferences(id=1, email="")
                obj.save()
            except IntegrityError:
                pass

        # Pass in context for rendered template
        context = {
            "m": m,
            "theme": theme.theme,
        }
        return render(request, "pages/dashboard.html", context)
    else:
        return redirect("/")


@login_required
def dashboard_settings_view(request):
    # Render dashboard settings template for matching url.

    # turn off action detection flag so it doesn't run in background
    detection.turn_off_detection()
    global model_settings
    if request.user.is_authenticated:
        # Get video settings from database
        video_settings = DashboardVideoSettings.objects.all()

        # Get model settings from database
        model_setting = DashboardModelSettings.objects.all()

        # Get email preference from database
        email_preference = EmailPreferences.objects.all()

        if request.method == "POST":
            # Update settings accordingly...

            # Uncheck all settings
            DashboardModelSettings.objects.all().order_by("id").update(switch=False)
            DashboardVideoSettings.objects.all().order_by("id").update(switch=False)

            # Set all global settings to false
            for key in model_settings:
                model_settings[key] = False

            # Get list of checkbed box id's
            video_settings_id_list = request.POST.getlist("dashboard_video_settings")
            model_settings_id_list = request.POST.getlist("dashboard_model_settings")

            # Get email preference and add it to the database
            email_update = request.POST.getlist("emails")
            try:
                if email_preference.count() > 1:
                    email_preference[0].delete()
                EmailPreferences.objects.filter(id=1).update(email=email_update[0])

            except IntegrityError:
                pass

            # Update the database
            for i in video_settings_id_list:
                DashboardVideoSettings.objects.filter(name_id=i).update(switch=True)
                model_settings[i] = True
            for i in model_settings_id_list:
                DashboardModelSettings.objects.filter(name_id=i).update(switch=True)
                model_settings[i] = True

            theme = Appearance.objects.get(appearance="theme")
            messages.success(request, "Settings updated successfully!")
            return render(
                request,
                "pages/settings.html",
                {
                    "video_settings": video_settings,
                    "model_settings": model_setting,
                    "theme": theme.theme,
                    "email_preference": email_preference[0].email,
                },
            )

        theme = Appearance.objects.get(appearance="theme")
        return render(
            request,
            "pages/settings.html",
            {
                "video_settings": video_settings,
                "model_settings": model_setting,
                "theme": theme.theme,
                "email_preference": email_preference[0].email,
            },
        )
    else:
        return redirect("/")


@login_required
def recordings_view(request):
    # Render dashboard recordings template for matching url.

    # turn off action detection flag so it doesn't run in background
    detection.turn_off_detection()
    theme = Appearance.objects.get(appearance="theme")
    recordings = Recordings.objects.all()
    context = {
        "theme": theme.theme,
        "recordings": recordings,
    }
    return render(request, "pages/recordings.html", context=context)


def security_logs(request):
    # An AJAX call retrieves the list of security threats and returns json.
    pacific_tz = pytz.timezone("US/Pacific")
    time_of_request = datetime.now(pacific_tz).strftime("%Y-%M-%d")
    data_for_request = {
        "time_of_request": time_of_request,
        "logs": loggers.security_notices,
        "whichlogs": "threat_log",
    }
    return JsonResponse(data_for_request)


def action_logs(request):
    # An AJAX call retrieves the list of model actions and returns json.
    pacific_tz = pytz.timezone("US/Pacific")
    time_of_request = datetime.now(pacific_tz).strftime("%Y-%M-%d")
    data_for_request = {
        "time_of_request": time_of_request,
        "logs": loggers.objects_detected,
        "whichlogs": "action_log",
    }
    return JsonResponse(data_for_request)

# pulls latitude and longitude from robot, then updates website map
def refresh_map_view(request):
    # An AJAX call to refresh the map on the dashboard home page.
    # Returns html representation of new map after a refresh click.

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

# publishes desired movement direction to robot
@csrf_exempt
def get_direction_data(request):
    if request.method == "POST":
        direction = request.POST["direction"]
        if direction == "N" or direction == "NE" or direction == "NW":
            pi_publisher.forward()
            # print("forward")
        elif direction == "S" or direction == "SE" or direction == "SW":
            pi_publisher.backward()
            # print("backward")
        elif direction == "W":
            pi_publisher.turn_left()
            # print("left")
        elif direction == "E":
            pi_publisher.turn_right()
            # print("right")

    return render(request, "pages/dashboard.html", {})

# publishes panning command to robot
@csrf_exempt
def get_panning_data(request):
    if request.method == "POST":
        panning = request.POST["panning"]
        if panning == "left":
            print("Left")
            pi_publisher.pan_left()
        if panning == "right":
            print("Right")
            pi_publisher.pan_right()
    return render(request, "pages/dashboard.html", {})

# updates global longitude and latitude variables and provides some console debugging
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

# publishes variable move distance to robot
@csrf_exempt
def send_distance(request):
    if request.method == "POST":
        try:
            distance = int(request.POST["distance"])
            if distance > 0 and distance <= 10:
                pi_publisher.move_distance(distance)
            else:
                print("error: send distance trying to send an invalid number")
        except ValueError:
            print("error: trying to send a non-int")
    return render(request, "pages/dashboard.html", {})


@csrf_exempt
def change_theme(request):
    # Used for a POST Call to change the theme to light/dark.
    # Updates the session variable in the database to light/dark,
    # This permits the theme to persist across sessions, changing from page to page constantly.
    t = Appearance.objects.get(appearance="theme")

    theme = request.POST["theme"]
    if theme == "light":
        t.theme = True  # light
    elif theme == "dark":
        t.theme = False  # dark

    t.save()

    response = {"theme": Appearance.objects.get(appearance="theme").theme}
    return JsonResponse(response)


def gen(url):
    global model_settings
    vcap = cv2.VideoCapture(url)
    while True:

        # Extract the names of the classes for trained the YoloV5 model
        class_ids = [0, 1, 2, 3]

        # Capture frame-by-frame
        ret, frame = vcap.read()

        # check if number of records is more than 9
        query_set = Recordings.objects.all()
        if query_set.count() > 9:
            query_set[0].delete()  # delete the oldest record in the table

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

            # if flag to run yolo is true apply
            if model_settings["Object Detection"] == True:

                # Apply the Torch YoloV5 model to this frame
                results = yolo(image)

                # Extract the labels and coordinates of the bounding boxes
                labels, cords = (
                    results.xyxyn[0][:, -1].numpy(),
                    results.xyxyn[0][:, :-1].numpy(),
                )

            # otherwise nothing was found in image
            else:
                labels = ""

            numberOfLabels = len(labels)

            # declare empty array of objects found
            objectsFound = []

            for i in range(numberOfLabels):
                # If global enable flag is set true then show boxes
                if model_settings["Bounding Box Overlay"] == True:
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
                            time_of_event = datetime.now(pacific_tz).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            seconds = int(datetime.today().timestamp() % 10)
                            if seconds == 0:
                                loggers.security_notices.append(
                                    [
                                        time_of_event,
                                        "Medium Security Alert (Malicious item detected): "
                                        + label,
                                    ]
                                )
                                # save current frame to a local file
                                cv2.imwrite("frame.jpg", frame)
                                # set current time to now
                                d = datetime.now(tz=pacific_tz)
                                # set description for the recording
                                description = "Malicious object detected"
                                # retrieve local file using Django's file reading
                                media = File(open("frame.jpg", "rb"))
                                # declare new record with this information and save to database
                                obj = Recordings(
                                    timestamp=d, description=description, media=media
                                )
                                obj.save()
                                # try to send email with notification
                                try:
                                    sendEmail()
                                except:
                                    print("Couldn't send email")

                        # append coords and label so it can be analyzed
                        objectsFound.append([x1, y1, x2, y2, label])

                        # send objected detected to log page
                        time_of_event = datetime.now(pacific_tz).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        seconds = int(datetime.today().timestamp() % 10)
                        if seconds == 0:
                            # every 10 seconds append to the log form
                            loggers.objects_detected.append(
                                [time_of_event, "Object detected: " + label]
                            )
                        if model_settings[label] == True:
                            # Draw bounding box
                            cv2.rectangle(
                                image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
                            )
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
                    if (
                        objectsFound[index][4] == "Angle Grinder"
                        or objectsFound[index][4] == "Bolt Cutters"
                    ):

                        # check if it is bike was detected in same frame
                        for index2 in range(len(objectsFound)):
                            if objectsFound[index2][4] == "Bike":
                                x1, y1, x2, y2, label1 = objectsFound[index]
                                x3, y3, x4, y4, label2 = objectsFound[index2]
                                box1 = [x1, y1, x2, y2]
                                box2 = [x3, y3, x4, y4]
                                # if interseciton is greater than 50 percent
                                # send a threat alert to alert logs
                                iou = iouCalc(box1, box2)
                                if iou >= 0.05:
                                    time_of_event = datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    )
                                    seconds = int(datetime.today().timestamp() % 10)

                                    loggers.security_notices.append(
                                        [
                                            time_of_event,
                                            "High Security Alert (malcious item on bike): "
                                            + label1
                                            + " near "
                                            + label2,
                                        ]
                                    )

            # return the resulting image
            _, jpeg = cv2.imencode(".jpg", image)
            frame = jpeg.tobytes()
            # return frame
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
            )


# function to send email to user's email
# code adapted from https://www.tutorialspoint.com/send-mail-from-your-gmail-account-using-python
def sendEmail():
    # define sending email
    emailSource = "patrolbotdash@gmail.com"
    # define sending password
    sourcePass = "dashPatrolsmtp5"
    # retrieve recipient email from database
    email_preference = EmailPreferences.objects.get(id=1)
    recipientEmail = email_preference.email

    # set up MIME
    message = MIMEMultipart()
    message["From"] = emailSource
    message["To"] = recipientEmail
    message["Subject"] = "PatrolBot Security Alert"

    # attach email content
    text = MIMEText("PatrolBot detected a malicious object. Check the logs now")
    message.attach(text)
    fp = open("frame.jpg", "rb")
    image = MIMEImage(fp.read(), name="MaliciousObject")
    message.attach(image)

    # ensure email is defined in database
    if recipientEmail != "":

        # create Gmail session
        session = smtplib.SMTP("smtp.gmail.com", 587, timeout=2)
        # add security
        session.starttls()
        # login
        session.login(emailSource, sourcePass)

        session.sendmail(emailSource, recipientEmail, message.as_string())
        print("Email sent")
        session.quit()


# code from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def iouCalc(boxA, boxB):
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
    # make sure previous threads are off
    actionDetectionOn = detection.get_flag_state()
    if actionDetectionOn == False:
        # create thread to run action detection
        thread = threading.Thread(target=detection.run_action_detection, args=(url,))
        # turn on action detection flag
        detection.turn_on_detection()
        thread.start()
    return StreamingHttpResponse(
        gen(url),
        content_type="multipart/x-mixed-replace;boundary=frame",
    )
