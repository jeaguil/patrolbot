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
    path("dashboard/recordings/", views.recordings_view, name="recordings"),
    path("kinesis_stream", views.kinesis_stream_view, name="kinesis_stream"),
    path("action_log_data", views.action_logs, name="action_log_data"),
    path("security_log_data", views.security_logs, name="security_log_data"),
    path("refresh_map", views.refresh_map_view, name="refresh_map"),
    path("get_direction", views.get_direction_data, name="get_direction"),
    path("get_panning", views.get_panning_data, name="get_panning"),
    path("change_theme", views.change_theme, name="change_theme"),
    path("send_coordinates", views.send_coordinates, name="send_coordinates"),
]
