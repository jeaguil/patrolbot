# Handles AWS MQTT subscribing to robot
import time
import os
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import json

# class-based subscriber implementation for easy declaration
class Subscriber():
    # set all necessary vars
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
        self.root_ca = root_ca_path
        self.key = private_key_path
        self.cert = certificate_path
        self._client = None
        self.finish = False
        self.daemon = True
        self.connected = False

    # connect to iot server
    def connect(self):
        self._client = AWSIoTMQTTClient(self.client_id)
        self._client.configureEndpoint(self.endpoint, 8883)
        self._client.configureCredentials(self.root_ca, self.key, self.cert)
        self._client.configureOfflinePublishQueueing(-1)
        self._client.configureConnectDisconnectTimeout(10)
        self._client.configureMQTTOperationTimeout(5)
        self.connected = self._client.connect()

    # subscribe to message
    def subscribe(self, topic, callback, qos=1):
        if not self.connected:
            self.connect()
        self._client.subscribe(topic, qos, callback)

    # keep running
    def run(self):
        while not self.finish:
            time.sleep(0.001)