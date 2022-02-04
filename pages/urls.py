from django.urls import path
from . import views

urlspatterns = [path("", views.welcome_view, name="welcome")]
