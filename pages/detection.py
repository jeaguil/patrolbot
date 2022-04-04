# import AWS SDK
import boto3

# import items for object detection
import cv2
import os
import torch
import pytz
import numpy as np

# import items for action detection
from mxnet import nd
from gluoncv.model_zoo import get_model
import torchvision.transforms as T

from django.conf import settings

# import items for updating action and security loggers
from . import loggers
from datetime import datetime

# name given to stream
STREAM_NAME = "MacStream"
AWS_REGION = "us-west-2"
# access key assigned to stream
ACCESS_KEY = "AKIAST56MMSDTPOIKMTM"
# secrety key assigned to stream
SECRET_KEY = "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"

# enable model flag
enable_model = True

model_weights = os.path.join(
    settings.BASE_DIR, "model_weights/patrolNanoWeights.pt"
)

action_weights = os.path.join(
    settings.BASE_DIR, "model_weights/ActionDetection.params"
)

pacific_tz = pytz.timezone('US/Pacific')

# variable to stop execution of the thread
runActionDetection = False


class KinesisStream(object):
    def __init__(self, url, yolo, colors):
        self.url = url
        self.yolo = yolo
        self.colors = colors

    def get_frame(self):
        return runYolo(self.url, self.yolo, self.colors)


# function code adapted
# from https://stackoverflow.com/questions/64030998/how-can-you-effectively-pull-an-amazon-kinesis-video-stream-for-custom-python-pr
def hls_stream():

    # Find the endpoint of the stream
    kv_client = boto3.client(
        "kinesisvideo",
        region_name=AWS_REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    endpoint = kv_client.get_data_endpoint(
        StreamName=STREAM_NAME, APIName="GET_HLS_STREAMING_SESSION_URL"
    )["DataEndpoint"]

    # print(endpoint)

    # Grab the HLS Stream URL from the endpoint
    # This URl has the location of the live stream
    kvam_client = boto3.client(
        "kinesis-video-archived-media",
        endpoint_url=endpoint,
        region_name=AWS_REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    url = kvam_client.get_hls_streaming_session_url(
        StreamName=STREAM_NAME, PlaybackMode="LIVE"
    )["HLSStreamingSessionURL"]

    # print(url)

    return url


# create yolo model and return it
def get_yolo():
    # Torch code adapted from https://github.com/akash-agni/Real-Time-Object-Detection/blob/main/Object_Detection_Youtube.py
    model_weight_path = model_weights
    model = torch.hub.load("ultralytics/yolov5", "custom", model_weight_path)
    return model


# create action model and return it
def get_action_model():
    # code adapted from https://cv.gluon.ai/build/examples_action_recognition/demo_slowfast_kinetics400.html
    # Load trained Slow Fast Model
    model_name = "slowfast_4x16_resnet50_kinetics400"
    #model_name = 'i3d_resnet50_v1_custom'
    net = get_model(model_name, nclass=400, pretrained=True)
    #net = get_model(model_name, nclass=2, pretrained = False)
    # net.load_parameters(action_weights)
    print("%s model is successfully loaded." % model_name)
    return net


def toggle_detection():
    global runActionDetection
    if runActionDetection == True:
        runActionDetection = False
    else:
        runActionDetection = True

# run action detection based on video stream from kinesis


def run_action_detection(url, net, request):
    # get list of frame numbers for fast portion of neural network
    fast_frame_id_list = range(0, 64, 2)
    # get list of frame numbers for slow portion of neural network
    slow_frame_id_list = range(0, 64, 16)
    # combine the two lists
    frame_id_list = list(fast_frame_id_list) + list(slow_frame_id_list)

    # frames captured initially set to 0
    frameCounter = 0
    # declare a list of frames
    clip_input = []
    # declare a list of slow frames
    slow_input = []

    # make a list of all potentially dangerous actions to detect
    dangerousActions = ['punching_bag', 'punching_person_-boxing-',
                        'wrestling', 'headbutting', 'drop_kicking', 'crying']
    vcap = cv2.VideoCapture(url)
    global runActionDetection
    while runActionDetection == True:

        # Capture frame-by-frame
        ret, frame = vcap.read()

        if ret:

            # Flip video frame, so it isn't reversed
            image = cv2.flip(frame, 1)

            # if the current frame was selected by the frame_id_list
            # add the frame to the list of frames
            if frameCounter in frame_id_list:
                # frame is originally 720 x 1280 x 3
                # form is height x width x dimensions
                # resize frame to required 224 x 224
                frame = cv2.resize(frame, (224, 224))
                # convert frame to tensor
                imgTensor = T.ToTensor()(frame)
                # create a transformation
                # normalize by mean and standard deviation
                transform = T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                )
                # apply transformation to tensor
                normalizedTensor = transform(imgTensor)
                # change tensor back to numpy array
                frame = normalizedTensor.cpu().detach().numpy()
                # frame has format 224 x 224 x 3
                # append the frame to the list of frames
                clip_input.append(frame)

                # if the frame is in the list of slow frame numbers
                # append to the slow input list
                if frameCounter in slow_frame_id_list:
                    slow_input.append(frame)

            # increment frame counter to keep track
            frameCounter += 1

            # if there are enough frames to run the action detection
            # apply it
            if frameCounter >= 70:
                # combine the video frames into one list
                # goes from 32 x 3 x 224 x 224 to 36 x 3 x 224 x 224
                clip_input = np.vstack((clip_input, slow_input))
                # join arrays on first axis
                clip_input = np.stack(clip_input, axis=0)
                # add a new dimensions and reshape
                clip_input = clip_input.reshape((-1,) + (36, 3, 224, 224))
                # finally array into 1 x 3 x 36 x 224 x 224
                # form is channels x frames x height x width
                clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

                # make the prediction based on the frames
                pred = net(nd.array(clip_input))

                actionClasses = net.classes
                #actionClasses = ['normal', 'abnormal']
                topK = 5
                ind = nd.topk(pred, k=topK)[0].astype("int")
                predictions = []

                # collect the top 5 predictions
                for i in range(topK):
                    predictions.append(
                        [
                            actionClasses[ind[i].asscalar()],
                            nd.softmax(pred)[0][ind[i]].asscalar(),
                        ]
                    )

                # extract the action with the highest confidence level
                bestPrediction = predictions[0]
                bestAction = bestPrediction[0]
                bestConfidence = bestPrediction[1]

                # if a dangerous action is detected
                if bestAction in dangerousActions and bestConfidence >= .5:
                    # send the alert to the alerts page
                    time_of_event = datetime.now(
                        pacific_tz).strftime("%Y-%m-%d %H:%M:%S")
                    seconds = int(datetime.today().timestamp() % 10)

                    loggers.security_notices.append(
                        [time_of_event, bestAction + " with confidence level of: " + str(bestConfidence)])

                # print out best action for stats
                print(bestAction, " with confidence ", bestConfidence)

                # reset frame counter to 0
                frameCounter = 0
                # empty the frames lists so they can be collected again
                # declare a list of frames
                clip_input = []
                # declare a list of slow frames
                slow_input = []
