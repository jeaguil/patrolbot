from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect
import folium
from . import loggers
from . import camera


def welcome_view(request):
    return render(request, "pages/welcome.html", {})


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
            "action_logger": loggers.action_logs,
            "security_logger": loggers.security_alerts_logs,
        }
        return render(request, "pages/dashboard.html", context)
    else:
        return redirect("/")


def dashboard_settings_view(request):
    return render(request, "pages/settings.html", {})


def dashboard_robot_view(request):
    return render(request, "pages/robot.html", {})


def dashboard_recordings_view(request):
    return render(request, "pages/recordings.html", {})


def gen(cam):
    while True:
        frame = cam.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


def phone_feed_view(request):
    return StreamingHttpResponse(
        gen(camera.IPPhoneCamera()),
        content_type="multipart/x-mixed-replace;boundary=frame",
    )
