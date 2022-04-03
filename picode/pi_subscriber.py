import time
import os
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import json

#root_ca_path1 = '/certificates/AmazonRootCA1.pem'
#private_key_path1 = '/certificates/private.pem.key'
#certificate_path1 = '/certificates/certificate.pem.crt'

#def gps_callback(self, params, packet):
#    payload = json.loads(packet.payload)
#    lat = payload["lat"]
#    lon = payload["lon"]

#def read_gps():
#    file_path = os.getcwd()
#    root_ca_path = file_path + root_ca_path1
#    private_key_path = file_path + private_key_path1
#    certificate_path = file_path + certificate_path1
#    myMQTTClient = AWSIoTMQTTClient("ServerClientID")##

#    myMQTTClient.configureEndpoint("aa03kvhkub5ls-ats.iot.us-west-2.amazonaws.com", 8883) #Provide your AWS IoT Core endpoint (Example: "abcdef12345-ats.iot.us-east-1.amazonaws.com")
#    myMQTTClient.configureCredentials(root_ca_path, private_key_path, certificate_path) #Set path for Root CA and provisioning claim credentials
#    myMQTTClient.configureOfflinePublishQueueing(-1)
#    myMQTTClient.configureDrainingFrequency(2)
#    myMQTTClient.configureConnectDisconnectTimeout(10)
#    myMQTTClient.configureMQTTOperationTimeout(5)
 
#    print("initiating iot core topic")
#    myMQTTClient.connect()

#    myMQTTClient.subscribe("robot/location", 1, gps_callback)
    #myMQTTClient.publish(
    #    topic="robot/control",
    #    QoS=1,
    #    payload='{"direction":"right"}'
    #)

class Subscriber():
    def __init__(self):
        root_ca_path1 = '/certificates/AmazonRootCA1.pem'
        private_key_path1 = '/certificates/private.pem.key'
        certificate_path1 = '/certificates/certificate.pem.crt'
        file_path = os.getcwd()
        root_ca_path = file_path + root_ca_path1
        private_key_path = file_path + private_key_path1
        certificate_path = file_path + certificate_path1
        self.endpoint = "aa03kvhkub5ls-ats.iot.us-west-2.amazonaws.com"
        self.client_id = "server"
        self.root_ca = root_ca_path#'/certificates/AmazonRootCA1.pem'
        self.key = private_key_path#'/certificates/private.pem.key'
        self.cert = certificate_path#'/certificates/certificate.pem.crt'
        self._client = None
        self.finish = False
        self.daemon = True
        self.connected = False

    def connect(self):
        self._client = AWSIoTMQTTClient(self.client_id)
        self._client.configureEndpoint(self.endpoint, 8883)
        self._client.configureCredentials(self.root_ca, self.key, self.cert)
        self._client.configureOfflinePublishQueueing(-1)
        self._client.configureConnectDisconnectTimeout(10)
        self._client.configureMQTTOperationTimeout(5)
        self.connected = self._client.connect()

    def subscribe(self, topic, callback, qos=1):
        if not self.connected:
            self.connect()
        self._client.subscribe(topic, qos, callback)

    def run(self):
        while not self.finish:
            time.sleep(0.001)