from django.contrib import admin
from .models import (
    DashboardVideoSettings,
    DashboardModelSettings,
    Appearance,
    Recordings,
    EmailPreferences,
)

for i in [
    DashboardVideoSettings,
    DashboardModelSettings,
    Appearance,
    Recordings,
    EmailPreferences,
]:
    admin.site.register(i)
