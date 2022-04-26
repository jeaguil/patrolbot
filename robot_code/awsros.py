# ROS control with AWS commands and GPS publishing

import time
import sys
import json
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from movement import MovementManager
from gps import *
import subprocess


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


def handle_control(self, params, packet):
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


robot = PatrolBotMovement()


def getLocation(gpsd):
    data = gpsd.next()
    if data['class'] == 'TPV':
        lon = getattr(data, 'lon', "Unknown")
        lat = getattr(data, 'lat', "Unknown")
        return lon, lat


active = True
gpsd = gps(mode=WATCH_ENABLE | WATCH_NEWSTYLE)

if __name__ == '__main__':
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

    active = True

    coordPublishTime = 0
    coordPublishDelay = 10 # in seconds

    gpsCheckTime = 0
    gpsCheckDelay = 2 # in seconds

    # handle gps publishing
    while(active):
        try:
            #while(active):
            if time.time() >= gpsCheckTime + gpsCheckDelay:
                coords = getLocation(gpsd)
                gpsCheckTime = time.time()
                #a = -119.807377167
                #b = 39.542380667
                #out = 
            # Handles faulty readings
            if time.time() >= coordPublishTime + coordPublishDelay and coords != None:
                if (coords == None):
                    print("No coords available")
                else:
                    #longitude = coords[0]
                    #latitude = coords[1]
                    #print("lon: {}, lat: {}".format(longitude, latitude))
                    filtered_coords = subprocess.check_output(args=["./run_filter.sh", "{} {}".format(coords[0], coords[1])])
                    #print(a)
                    #print(type(a))
                    filtered_coords = tuple(filtered_coords.split (","))
                    longitude = filtered_coords[0]
                    latitude = filtered_coords[1]
                    #print("formatted: {}, {}".format(filtered_coords[0], filtered_coords[1]))
                    #print("og long {} og lat {}".format(coords[0], coords[1]))

                    TOPIC = "robot/location"
                    MESSAGE1 = str(longitude)# + "," + str(latitude)
                    MESSAGE2 = str(latitude)
                    data1 = "{}".format(MESSAGE1)
                    data2 = "{}".format(MESSAGE2)
                    message = {"lon" : data1, "lat" : data2}
                    myMQTTClient.publish(TOPIC, json.dumps(message), 1)
                    print("Printed '" + json.dumps(message) + "' to the topic: " + TOPIC)
                    coordPublishTime = time.time()
                    #adjust sleep time for frequency of readings


        except (KeyboardInterrupt):
            active = False
            print("Exiting")

        #except (AWSIoTPythonSDK.exception.AWSIoTExceptions.publishTimeoutException):
        #    print("aws iot python sdk exception raised, giving robot a break")
        #    time.sleep(5)
        #    pass
