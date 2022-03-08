from datetime import datetime

time_of_event = datetime.now()
current_time = time_of_event.strftime("%H:%M:%S")

""" DEFAULT LOG DATA 

    Before the model begins obj detection or computes potential
    security threats, this is the default heading for the log data."""

action_logs = [
    "No objects detected",
    current_time + " - " + "Object detected",
]
security_alerts_logs = [
    "No threats compromised",
    current_time + " - " + "Threat compromised",
]
