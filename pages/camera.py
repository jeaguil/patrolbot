import cv2
import os
import urllib.request
import numpy as np
import datetime
import threading

from django.conf import settings

# Tempory model data for IP phone camera tests
face_detect_cam = cv2.CascadeClassifier(
    os.path.join(
        settings.BASE_DIR, "opencv_haarcascade_data/haarcascade_frontalface_data.xml"
    )
)

log_objs = True


# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)

#     def __del__(self):
#         self.video.release()

#     def get_frame(self):
#         success, image = self.video.read()

#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         faces_detected = face_detection_videocam.detectMultiScale(
#             gray, scaleFactor=1.3, minNeighbors=5
#         )
#         for (x, y, w, h) in faces_detected:
#             cv2.rectangle(
#                 image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2
#             )
#         frame_flip = cv2.flip(image, 1)
#         ret, jpeg = cv2.imencode(".jpg", frame_flip)
#         return jpeg.tobytes()


class IPPhoneCamera(object):
    """For testing purposes.

    Attempts to stream live feed provided by the IP camera app on a phone
    to the web page.

    Not expected to work in production."""

    def __init__(self):

        """IP Webcam URL
        ->Start Server
        ->IPv4 address
        ->Video renderer in javascript
        ->Copy img link"""
        self.url = "http://192.168.1.234:8080/shot.jpg"

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        image_response = urllib.request.urlopen(self.url)

        image_array = np.array(bytearray(image_response.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, 1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_detected = face_detect_cam.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # update_logs("obj detected")

        resize = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        frame_flip = cv2.flip(resize, 1)
        ret, jpeg = cv2.imencode(".jpg", frame_flip)
        return jpeg.tobytes()


def update_logs(notice: str):
    threading.Timer(10.0, update_logs).start()
    time_of_event = datetime.now()
    new_log = time_of_event.strftime("%H:%M:%S") + notice
    action_logs += new_log
