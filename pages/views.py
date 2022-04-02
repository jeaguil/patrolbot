from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import DashboardModelSettings, DashboardVideoSettings
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
    new_robot_location = [48.864716, 2.349014]
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
    if request.method == "POST":
        direction = request.POST["direction"]
        if direction == "N" or direction == "NE" or direction == "NW":
            print("forward")
        elif direction == "S" or direction == "SE" or direction == "SW":
            print("backward")
        elif direction == "W":
            print("left")
        elif direction == "E":
            print("right")
        else:
            print("No Movement")
        return render(request, "pages/dashboard.html", {})


@login_required
def dashboard_view(request):
    if request.user.is_authenticated:
        robo_coords = [39.54244129476235, -119.81597984878438]

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

        # Pass in context for rendered template
        context = {
            "m": m,
        }
        return render(request, "pages/dashboard.html", context)
    else:
        return redirect("/")


@login_required
def dashboard_settings_view(request):
    if request.user.is_authenticated:
        # Get video settings from database
        video_settings = DashboardVideoSettings.objects.all()

        # Get model settings from database
        model_settings = DashboardModelSettings.objects.all()

        if request.method == "POST":
            # Update settings accordingly...

            # Uncheck all settings
            DashboardModelSettings.objects.all().order_by("id").update(switch=False)
            DashboardVideoSettings.objects.all().order_by("id").update(switch=False)

            # Get list of checkbed box id's
            video_settings_id_list = request.POST.getlist("dashboard_video_settings")
            model_settings_id_list = request.POST.getlist("dashboard_model_settings")

            # Update the database
            for i in video_settings_id_list:
                DashboardVideoSettings.objects.filter(name_id=i).update(switch=True)
            for i in model_settings_id_list:
                DashboardModelSettings.objects.filter(name_id=i).update(switch=True)

            messages.success(request, "Settings updated successfully!")
            return render(
                request,
                "pages/settings.html",
                {"video_settings": video_settings, "model_settings": model_settings},
            )

        return render(
            request,
            "pages/settings.html",
            {"video_settings": video_settings, "model_settings": model_settings},
        )
    else:
        return redirect("/")


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
