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
]
