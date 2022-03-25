from django.db import models

class DashboardVideoSettings(models.Model):
    name_id = models.CharField(max_length=100, blank=False, default='', unique=True)
    setting = models.CharField(max_length=100, blank=False, default='', unique=True)
    switch = models.BooleanField('switch', default=True)
    
    def __str__(self):
        return self.name_id
    
class DashboardModelSettings(models.Model):
    name_id = models.CharField(max_length=100, blank=False, default='', unique=True)
    setting = models.CharField(max_length=100, blank=False, default='', unique=True)
    switch = models.BooleanField('switch', default=True)
    
    def __str__(self):
        return self.name_id