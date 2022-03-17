from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
import folium
import threading
import pytz
from datetime import datetime
import numpy as np
from . import loggers
from . import camera

# mxnet and gluoncv must be built from source
# install CPU version of mxnet and gluoncv before running this
from . import detection

from picode import pi_publisher

def welcome_view(request):
    return render(request, "pages/welcome.html", {})

def security_logs(request):
    pacific_tz = pytz.timezone('US/Pacific')
    time_of_request = datetime.now(pacific_tz).strftime("%Y-%M-%d")
    data_for_request = {
        "time_of_request": time_of_request,
        "logs": loggers.security_notices,
        "whichlogs": "threat_log",
    }
    return JsonResponse(data_for_request)

def action_logs(request):
    pacific_tz = pytz.timezone('US/Pacific')
    time_of_request = datetime.now(pacific_tz).strftime("%Y-%M-%d")
    data_for_request = {
        "time_of_request": time_of_request,
        "logs": loggers.objects_detected,
        "whichlogs": "action_log",
    }
    return JsonResponse(data_for_request)


@login_required
def dashboard_view(request):
    if request.user.is_authenticated:
        robo_coords = [39.54244129476235, -119.81597984878438]

        f = folium.Figure(width="100%", height=700)
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

        # render map in context for template
        context = {
            "m": m,
        }
        return render(request, "pages/dashboard.html", context)
    else:
        return redirect("/")


def dashboard_settings_view(request):
    return render(request, "pages/settings.html")


def dashboard_robot_view(request):
    return render(request, "pages/robot.html", {})


def dashboard_recordings_view(request):
    print("recordings view")
    return render(request, "pages/recordings.html", {})


def dashboard_robot_manual_view(request):
    if request.method == "POST":
        if "forward_command" in request.POST:
            pi_publisher.forward()
        elif "backward_command" in request.POST:
            pi_publisher.backward()
        elif "turn_left_command" in request.POST:
            pi_publisher.turn_left()
        elif "turn_right_command" in request.POST:
            pi_publisher.turn_right()
    return render(request, "pages/robot_manual.html", {})


def gen(cam):
    while True:
        frame = cam.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


def phone_feed_view(request):
    return StreamingHttpResponse(
        gen(camera.IPPhoneCamera()),
        content_type="multipart/x-mixed-replace;boundary=frame",
    )


def kinesis_stream_view(request):
    # retrieves url on hls stream
    url = detection.hls_stream()
    # create yolo model
    yolo = detection.get_yolo()
    classes = yolo.names
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    # create action detection model
    action = detection.get_action_model()
    # create thread to run action detection
    thread = threading.Thread(target=detection.run_action_detection, args=(url, action))
    thread.start()
    return StreamingHttpResponse(
        gen(detection.KinesisStream(url, yolo, COLORS)),
        content_type="multipart/x-mixed-replace;boundary=frame",
    )
