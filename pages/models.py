from django.db import models


class Action(models.Model):
    time_of_event = models.DateTimeField(auto_now_add=True)
    object_detected = models.CharField(max_length=100)


class SecurityThreat(models.Model):
    time_of_event = models.DateField(auto_now_add=True)
    type_of_threat = models.CharField(max_length=100)
