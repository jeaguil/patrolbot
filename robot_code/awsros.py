# ROS control with AWS commands and GPS publishing

import time
import sys
import json
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from movement import MovementManager
from gps import *

class PatrolBotMovement:
	def __init__(self):
		self.movement_manager = MovementManager()
	
	def forward(self):
		self.movement_manager.set_max_linear_velocity(0.5)
		# move forward 1 meter
		self.movement_manager.move_straight(0.25)

	def backward(self):
		self.movement_manager.set_max_linear_velocity(0.5)
		self.movement_manager.move_straight(-0.25)

	def rotateClockwise(self):
		self.movement_manager.set_max_angular_velocity(0.5)
		self.movement_manager.turn(-1.35)

	def rotateCounterClockwise(self):
		self.movement_manager.set_max_angular_velocity(0.5)
		self.movement_manager.turn(1.35)

#class MsgSys:
#    #robot = PatrolBotMovement()
    
#    def __init__(self):
#        self.myMQTTClient = AWSIoTMQTTClient("clientid")
#        self.myMQTTClient.configureEndpoint("aa03kvhkub5ls-ats.iot.us-west-2.amazonaws.com", 8883)
#        self.myMQTTClient.configureCredentials("/home/ubuntu/AWSIoT/AmazonRootCA1.pem", "/home/ubuntu/AWSIoT/private.pem.key", "/home/ubuntu/AWSIoT/certificate.pem.crt") #Set path for Root CA and provisioning claim credentials
#        self.myMQTTClient.configureOfflinePublishQueueing(-1)
#        self.myMQTTClient.configureDrainingFrequency(2)
#        self.myMQTTClient.configureConnectDisconnectTimeout(10)
#        self.myMQTTClient.configureMQTTOperationTimeout(5)
#        print ('Initiating IoT Core Topic ...')
#        self.myMQTTClient.connect()
#        self.myMQTTClient.subscribe("robot/control", 1, handle_control)
        
def handle_control(self, params, packet):
    payload = json.loads(packet.payload)
    command = payload["direction"]
    print(command)
    if command == 'forward':
        print('moving forward')
        robot.forward()
    elif command == 'backward':
        print('moving backward')
        robot.backward()
    elif command == 'left':
        print('rotating left')
        robot.rotateClockwise()
    elif command == 'right':
        print('rotating right')
        robot.rotateCounterClockwise()
    else:
        print('error: unknown command')

robot = PatrolBotMovement()

def getLocation(gpsd):   
    data = gpsd.next()
    if data['class'] == 'TPV':
        lon = getattr(data, 'lon', "Unknown")
        lat = getattr(data, 'lat', "Unknown")
        return lon, lat

active = True
gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)

if __name__ == '__main__':
    #msg = MsgSys()
    myMQTTClient = AWSIoTMQTTClient("clientid")
    myMQTTClient.configureEndpoint("aa03kvhkub5ls-ats.iot.us-west-2.amazonaws.com", 8883)
    myMQTTClient.configureCredentials("/home/ubuntu/patrolbot/certificates/AmazonRootCA1.pem", "/home/ubuntu/patrolbot/certificates/private.pem.key", "/home/ubuntu/patrolbot/certificates/certificate.pem.crt") #Set path for Root CA and provisioning claim credentials
    myMQTTClient.configureOfflinePublishQueueing(-1)
    myMQTTClient.configureDrainingFrequency(2)
    myMQTTClient.configureConnectDisconnectTimeout(10)
    myMQTTClient.configureMQTTOperationTimeout(5)
    print ('Initiating IoT Core Topic ...')
    myMQTTClient.connect()
    myMQTTClient.subscribe("robot/control", 1, handle_control)
    
    active = True
    # handle gps publishing
    try:
        while(active):
        #time.sleep(1)    
            coords = getLocation(gpsd)
        
        #Handles faulty readings
            if (coords == None):
                print("No coords available!")
                pass
            else:
                longitude = coords[0]
                latitude = coords[1]
                print("latlong:" + str(longitude) + "," + str(latitude))
                TOPIC = "robot/location"
                MESSAGE1 = str(longitude)# + "," + str(latitude)
		MESSAGE2 = str(latitude)
                data1 = "{}".format(MESSAGE1)
		data2 = "{}".format(MESSAGE2)
                message = {"lon" : data1, "lat" : data2}
                myMQTTClient.publish(TOPIC, json.dumps(message), 1)
                print("Printed '" + json.dumps(message) + "' to the topic: " + TOPIC)
                #adjust sleep time for frequency of readings
                time.sleep(10)
            #myMQTTClient.publish(
            #    topic="robot/location",
            #    QoS=1,
            #    payload='{"direction":"right"}'
            #)

        
    except (KeyboardInterrupt):
        active = False
        print("Exiting") 
