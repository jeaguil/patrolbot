from django.contrib import admin
from .models import DashboardVideoSettings, DashboardModelSettings, Appearance

# Register your models here.
for i in [DashboardVideoSettings, DashboardModelSettings, Appearance]:
    admin.site.register(i)