from django.urls import path
from . import views

urlpatterns = [
    path("", views.welcome_view, name="welcome"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path(
        "dashboard/settings/",
        views.dashboard_settings_view,
        name="settings",
    ),
    path(
        "dashboard/robot/",
        views.dashboard_robot_view,
        name="robot",
    ),
    path("dashboard/recordings/", views.dashboard_recordings_view, name="recordings"),
    path(
        "dashboard/robot_manual/",
        views.dashboard_robot_manual_view,
        name="robot_manual",
    ),
    # camera urls->
    path("phone_feed", views.phone_feed_view, name="phone_feed"),
    path("kinesis_stream", views.kinesis_stream_view, name="kinesis_stream"),
    path("logdata", views.logdata_view, name="logdata"),
]
