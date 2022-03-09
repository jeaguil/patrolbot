import time
import os
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

root_ca_path1 = '/certificates/AmazonRootCA1.pem'
private_key_path1 = '/certificates/private.pem.key'
certificate_path1 = '/certificates/certificate.pem.crt'

def forward():
    file_path = os.getcwd()
    root_ca_path = file_path + root_ca_path1
    private_key_path = file_path + private_key_path1
    certificate_path = file_path + certificate_path1
    myMQTTClient = AWSIoTMQTTClient("ServerClientID")

    myMQTTClient.configureEndpoint("aa03kvhkub5ls-ats.iot.us-west-2.amazonaws.com", 8883) #Provide your AWS IoT Core endpoint (Example: "abcdef12345-ats.iot.us-east-1.amazonaws.com")
    myMQTTClient.configureCredentials(root_ca_path, private_key_path, certificate_path) #Set path for Root CA and provisioning claim credentials
    myMQTTClient.configureOfflinePublishQueueing(-1)
    myMQTTClient.configureDrainingFrequency(2)
    myMQTTClient.configureConnectDisconnectTimeout(10)
    myMQTTClient.configureMQTTOperationTimeout(5)
 
    print("initiating iot core topic")
    myMQTTClient.connect()
    myMQTTClient.publish(
        topic="robot/control",
        QoS=1,
        payload='{"direction":"forward"}'
    )

def backward():
    file_path = os.getcwd()
    root_ca_path = file_path + root_ca_path1
    private_key_path = file_path + private_key_path1
    certificate_path = file_path + certificate_path1
    myMQTTClient = AWSIoTMQTTClient("ServerClientID")

    myMQTTClient.configureEndpoint("aa03kvhkub5ls-ats.iot.us-west-2.amazonaws.com", 8883) #Provide your AWS IoT Core endpoint (Example: "abcdef12345-ats.iot.us-east-1.amazonaws.com")
    myMQTTClient.configureCredentials(root_ca_path, private_key_path, certificate_path) #Set path for Root CA and provisioning claim credentials
    myMQTTClient.configureOfflinePublishQueueing(-1)
    myMQTTClient.configureDrainingFrequency(2)
    myMQTTClient.configureConnectDisconnectTimeout(10)
    myMQTTClient.configureMQTTOperationTimeout(5)
 
    print("initiating iot core topic")
    myMQTTClient.connect()
    myMQTTClient.publish(
        topic="robot/control",
        QoS=1,
        payload='{"direction":"backward"}'
    )

def turn_left():
    file_path = os.getcwd()
    root_ca_path = file_path + root_ca_path1
    private_key_path = file_path + private_key_path1
    certificate_path = file_path + certificate_path1
    myMQTTClient = AWSIoTMQTTClient("ServerClientID")

    myMQTTClient.configureEndpoint("aa03kvhkub5ls-ats.iot.us-west-2.amazonaws.com", 8883) #Provide your AWS IoT Core endpoint (Example: "abcdef12345-ats.iot.us-east-1.amazonaws.com")
    myMQTTClient.configureCredentials(root_ca_path, private_key_path, certificate_path) #Set path for Root CA and provisioning claim credentials
    myMQTTClient.configureOfflinePublishQueueing(-1)
    myMQTTClient.configureDrainingFrequency(2)
    myMQTTClient.configureConnectDisconnectTimeout(10)
    myMQTTClient.configureMQTTOperationTimeout(5)
 
    print("initiating iot core topic")
    myMQTTClient.connect()
    myMQTTClient.publish(
        topic="robot/control",
        QoS=1,
        payload='{"direction":"left"}'
    )

def turn_right():
    file_path = os.getcwd()
    root_ca_path = file_path + root_ca_path1
    private_key_path = file_path + private_key_path1
    certificate_path = file_path + certificate_path1
    myMQTTClient = AWSIoTMQTTClient("ServerClientID")

    myMQTTClient.configureEndpoint("aa03kvhkub5ls-ats.iot.us-west-2.amazonaws.com", 8883) #Provide your AWS IoT Core endpoint (Example: "abcdef12345-ats.iot.us-east-1.amazonaws.com")
    myMQTTClient.configureCredentials(root_ca_path, private_key_path, certificate_path) #Set path for Root CA and provisioning claim credentials
    myMQTTClient.configureOfflinePublishQueueing(-1)
    myMQTTClient.configureDrainingFrequency(2)
    myMQTTClient.configureConnectDisconnectTimeout(10)
    myMQTTClient.configureMQTTOperationTimeout(5)
 
    print("initiating iot core topic")
    myMQTTClient.connect()
    myMQTTClient.publish(
        topic="robot/control",
        QoS=1,
        payload='{"direction":"right"}'
    )