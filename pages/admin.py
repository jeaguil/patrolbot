from django.contrib import admin
from .models import DashboardVideoSettings, DashboardModelSettings, Appearance, Recordings

# Register your models here.
for i in [DashboardVideoSettings, DashboardModelSettings, Appearance, Recordings]:
    admin.site.register(i)