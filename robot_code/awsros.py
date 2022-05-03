# ROS control with AWS commands and GPS publishing

import time
import sys
import json
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from movement import MovementManager
from gps import *
import subprocess

# robot movement control class, connects with ros in movement.py
class PatrolBotMovement:
    def __init__(self):
        self.movement_manager = MovementManager()

    def forward(self):
        self.movement_manager.set_max_linear_velocity(0.5)
        distanceMoved = self.movement_manager.move_straight(0.25)
        return distanceMoved

    def backward(self):
        self.movement_manager.set_max_linear_velocity(0.5)
        distanceMoved = self.movement_manager.move_straight(-0.25)
        return distanceMoved

    def rotateClockwise(self):
        self.movement_manager.set_max_angular_velocity(0.5)
        distanceMoved = self.movement_manager.turn(-1.35)
        return distanceMoved

    def rotateCounterClockwise(self):
        self.movement_manager.set_max_angular_velocity(0.5)
        distanceMoved = self.movement_manager.turn(1.35)
        return distanceMoved

    def panLeft(self):
        self.movement_manager.set_max_angular_velocity(0.5)
        self.movement_manager.turn(1.35)
        time.sleep(2)
        self.movement_manager.turn(-1.35)
        return 1

    def panRight(self):
        self.movement_manager.set_max_angular_velocity(0.5)
        self.movement_manager.turn(-1.35)
        time.sleep(2)
        self.movement_manager.turn(1.35)
        return 1

# aws subscriber callback handles calling robot functions
def handle_control(self, params, packet):
    # get packet from aws iot server
    payload = json.loads(packet.payload)
    if "direction" in payload:
        moveCommand = payload["direction"]
        if moveCommand == 'forward':
            print("moving forward")
            robot.forward()
            print("movement complete")
        elif moveCommand == 'backward':
            print("moving backward")
            robot.backward()
            print("movement complete")
        elif moveCommand == 'left':
            print("moving left")
            robot.rotateCounterClockwise()
            print("movement complete")
        elif moveCommand == 'right':
            print("moving right")
            robot.rotateClockwise()
            print("movement complete")
        else:
            print("Unknown direction command")
    elif "distance" in payload:
        desiredDistance = int(payload["distance"])
        complete = False
        distanceTraveled = 0
        while complete is not True:
            distanceTraveled += robot.forward()
            time.sleep(2)
            if distanceTraveled >= desiredDistance:
                complete = True
    elif "camera" in payload:
        cameraDirection = payload["camera"]
        if cameraDirection == 'left':
            robot.panLeft()
            print("panning left")
        elif cameraDirection == 'right':
            print("panning right")
            robot.panRight()
        else:
            print("Unknown panning direction")
    else:
        print("Unknown key in payload")

# global robot declaration
robot = PatrolBotMovement()

# handle gathering gps coordinates
def getLocation(gpsd):
    data = gpsd.next()
    if data['class'] == 'TPV':
        lon = getattr(data, 'lon', "Unknown")
        lat = getattr(data, 'lat', "Unknown")
        return lon, lat

# set global active variable for easy closing
active = True
# initialize gps modes
gpsd = gps(mode=WATCH_ENABLE | WATCH_NEWSTYLE)

if __name__ == '__main__':
    # connect to aws server
    myMQTTClient = AWSIoTMQTTClient("clientid")
    myMQTTClient.configureEndpoint(
        "aa03kvhkub5ls-ats.iot.us-west-2.amazonaws.com", 8883)
    myMQTTClient.configureCredentials("/home/ubuntu/patrolbot/certificates/AmazonRootCA1.pem", "/home/ubuntu/patrolbot/certificates/private.pem.key",
                                      "/home/ubuntu/patrolbot/certificates/certificate.pem.crt")  # Set path for Root CA and provisioning claim credentials
    myMQTTClient.configureOfflinePublishQueueing(-1)
    myMQTTClient.configureDrainingFrequency(2)
    myMQTTClient.configureConnectDisconnectTimeout(10)
    myMQTTClient.configureMQTTOperationTimeout(5)
    print('Initiating IoT Core Topic ...')
    myMQTTClient.connect()
    myMQTTClient.subscribe("robot/control", 1, handle_control)

    # create delay and time vars for timed command control
    coordPublishTime = 0
    coordPublishDelay = 10 # in seconds

    gpsCheckTime = 0
    gpsCheckDelay = 2 # in seconds

    # handle gps publishing
    while(active):
        try:
            if time.time() >= gpsCheckTime + gpsCheckDelay:
                coords = getLocation(gpsd) # get coords, dont check again for gpsCheckDelay seconds
                gpsCheckTime = time.time()
            # Handles faulty readings
            if time.time() >= coordPublishTime + coordPublishDelay and coords != None:
                if (coords == None):
                    print("No coords available")
                else:
                    # run filter, get output
                    filtered_coords = subprocess.check_output(args=["./run_filter.sh", "{} {}".format(coords[0], coords[1])])
                    # parse filtered string
                    filtered_coords = tuple(filtered_coords.split (","))
                    longitude = filtered_coords[0]
                    latitude = filtered_coords[1]

                    # establish data for publishing
                    TOPIC = "robot/location"
                    MESSAGE1 = str(longitude)
                    MESSAGE2 = str(latitude)
                    data1 = "{}".format(MESSAGE1)
                    data2 = "{}".format(MESSAGE2)
                    message = {"lon" : data1, "lat" : data2}
                    # publish to server
                    myMQTTClient.publish(TOPIC, json.dumps(message), 1)
                    print("Printed '" + json.dumps(message) + "' to the topic: " + TOPIC)
                    coordPublishTime = time.time()

        # if interrupt pressed, exit project
        except (KeyboardInterrupt):
            active = False
            print("Exiting")
