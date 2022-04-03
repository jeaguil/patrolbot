# Publish data to robot using MQTT

#import time
#from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

def helloworld(self, params, packet):
    print ('Recieved Message from AWS IoT Core')
    print ('Topic: ' + packet.topic)
    print ("Payload: ", (packet.payload))

myMQTTClient = AWSIoTMQTTClient("clientid")
myMQTTClient.configureEndpoint("aa03kvhkub5ls-ats.iot.us-west-2.amazonaws.com", 8883)

myMQTTClient.configureCredentials("/home/ubuntu/AWSIoT/AmazonRootCA1.pem", "/home/ubuntu/AWSIoT/private.pem.key", "/home/ubuntu/AWSIoT/certificate.pem.crt") #Set path for Root CA and provisioning claim credentials

myMQTTClient.configureOfflinePublishQueueing(-1)
myMQTTClient.configureDrainingFrequency(2)
myMQTTClient.configureConnectDisconnectTimeout(10)
myMQTTClient.configureMQTTOperationTimeout(5)
 
print ('Initiating IoT Core Topic ...')
myMQTTClient.connect()

#myMQTTClient.subscribe("home/helloworld", 1, helloworld)

#while True:
#	time.sleep(5)

print("Publishing Message from RPI")
myMQTTClient.publish(
	topic="home/helloworld",
	QoS=1,
	payload="{'Message':'Message by RPI'}")


def move(direction):
    if(direction == 'forward'):
        print("forward")
    elif(direction == 'backward'):
        print("backward")
    elif(direction == 'left'):
        print("left")
    elif(direction == 'right'):
        print("right")
