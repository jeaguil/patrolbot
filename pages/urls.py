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
    #path(
    #    "dashboard/robot_manual/",
    #    views.dashboard_robot_manual_view,
    #    name="robot_manual",
    #),
    path("phone_feed", views.phone_feed_view, name="phone_feed"),
    path("kinesis_stream", views.kinesis_stream_view, name="kinesis_stream"),
    path("action_log_data", views.action_logs, name="action_log_data"),
    path("security_log_data", views.security_logs, name="security_log_data"),
    path("model_meta_data", views.model_meta_data, name="model_meta_data"),
    path("refresh_map", views.refresh_map_view, name="refresh_map"),
    path("get_direction", views.get_direction_data, name="get_direction"),
    path("change_theme", views.change_theme, name="change_theme")
]