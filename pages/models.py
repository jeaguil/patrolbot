from django.db import models

class DashboardVideoSettings(models.Model):
    setting = models.CharField(max_length=100)
    switch = models.BooleanField('switch', default=True)
    
    def __str__(self):
        return self.setting
    
class DashboardModelSettings(models.Model):
    setting = models.CharField(max_length=100)
    switch = models.BooleanField('switch', default=True)
    
    def __str__(self):
        return self.setting