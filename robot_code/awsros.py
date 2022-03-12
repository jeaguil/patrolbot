# ROS control with AWS commands

import time
import sys
import json
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from movement import MovementManager

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

class MsgSys:
    #robot = PatrolBotMovement()
    
    def __init__(self):
        self.myMQTTClient = AWSIoTMQTTClient("clientid")
        self.myMQTTClient.configureEndpoint("aa03kvhkub5ls-ats.iot.us-west-2.amazonaws.com", 8883)
        self.myMQTTClient.configureCredentials("/home/ubuntu/AWSIoT/AmazonRootCA1.pem", "/home/ubuntu/AWSIoT/private.pem.key", "/home/ubuntu/AWSIoT/certificate.pem.crt") #Set path for Root CA and provisioning claim credentials
        self.myMQTTClient.configureOfflinePublishQueueing(-1)
        self.myMQTTClient.configureDrainingFrequency(2)
        self.myMQTTClient.configureConnectDisconnectTimeout(10)
        self.myMQTTClient.configureMQTTOperationTimeout(5)
        print ('Initiating IoT Core Topic ...')
        self.myMQTTClient.connect()
        self.myMQTTClient.subscribe("robot/control", 1, handle_control)
        
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

if __name__ == '__main__':
    msg = MsgSys()
    while(True):
        time.sleep(1) 