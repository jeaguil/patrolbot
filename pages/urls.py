from django.urls import path
from . import views

urlpatterns = [
    path("", views.welcome_view, name="welcome"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("dashboard/settings/", views.dashboard_settings_view, name="settings",),
    path("dashboard/robot_manual/", views.dashboard_robot_manual_view, name="robot_manual"),
    path("phone_feed", views.phone_feed_view, name="phone_feed"),
    path("kinesis_stream", views.kinesis_stream_view, name="kinesis_stream"),
    path("action_log_data", views.action_logs, name="action_log_data"),
    path("security_log_data", views.security_logs, name="security_log_data"),
]
