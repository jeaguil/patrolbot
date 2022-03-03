# import AWS SDK
import boto3

# import items for object detection
import cv2
import os
import torch
import numpy as np

# import items for action detection
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model
import torchvision.transforms as T

# name given to stream
STREAM_NAME = "MacStream"
AWS_REGION = 'us-east-1'
# access key assigned to stream
ACCESS_KEY = 'AKIAST56MMSDTPOIKMTM'
# secrety key assigned to stream
SECRET_KEY = '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'

# function code adapted 
# from https://stackoverflow.com/questions/64030998/how-can-you-effectively-pull-an-amazon-kinesis-video-stream-for-custom-python-pr
def hls_stream():

    # Find the endpoint of the stream
    kv_client = boto3.client("kinesisvideo", region_name=AWS_REGION, aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY)
    endpoint = kv_client.get_data_endpoint(
        StreamName=STREAM_NAME,
        APIName="GET_HLS_STREAMING_SESSION_URL"
    )['DataEndpoint']

    #print(endpoint)

    # Grab the HLS Stream URL from the endpoint
    # This URl has the location of the live stream
    kvam_client = boto3.client("kinesis-video-archived-media", endpoint_url=endpoint, region_name=AWS_REGION, aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY)
    url = kvam_client.get_hls_streaming_session_url(
        StreamName=STREAM_NAME,
        PlaybackMode="LIVE"
    )['HLSStreamingSessionURL']

    #print(url)

    return url

# run yolo and action detection based on video stream from kinesis
def runModels(url):
    #vcap = cv2.VideoCapture(url)
    vcap = cv2.VideoCapture(0)

    # Torch code adapted from https://github.com/akash-agni/Real-Time-Object-Detection/blob/main/Object_Detection_Youtube.py
    model_weight_path = os.path.join(os.getcwd(), 'Documents/CS426/PatrolBot/model_weights/best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', model_weight_path)
    
    # Extract the names of the classes for trained the YoloV5 model
    classes = model.names
    class_ids = [0,1,2,3]
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

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

    # code adapted from https://cv.gluon.ai/build/examples_action_recognition/demo_slowfast_kinetics400.html
    # Load trained Slow Fast Model
    model_name = 'slowfast_4x16_resnet50_kinetics400'
    net = get_model(model_name, nclass=400, pretrained=True)
    print('%s model is successfully loaded.' % model_name) 

    while(True):
        # Capture frame-by-frame
        ret, frame = vcap.read()

        if ret:

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip video frame, so it isn't reversed
            image = cv2.flip(image, 1)

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
                transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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
                clip_input = clip_input.reshape((-1,) + (36,3,224,224))
                # finally array into 1 x 3 x 36 x 224 x 224
                # form is channels x frames x height x width
                clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

                # make the prediction based on the frames
                pred = net(nd.array(clip_input))

                actionClasses = net.classes
                topK = 5
                ind = nd.topk(pred, k=topK)[0].astype('int')
                predictions = []

                # collect the top 5 predictions
                for i in range(topK):
                    predictions.append([actionClasses[ind[i].asscalar()],nd.softmax(pred)[0][ind[i]].asscalar()])

                # extract the action with the highest confidence level
                bestPrediction=predictions[0]
                bestAction = bestPrediction[0]
                bestConfidence = bestPrediction[1]

                # print out best action for stats
                #print(bestAction, " with confidence ", bestConfidence)

                # reset frame counter to 0
                frameCounter = 0
                # empty the frames lists so they can be collected again
                # declare a list of frames
                clip_input = []
                # declare a list of slow frames
                slow_input = []

            # If model is turned on and the object is initialized
            # run object detection on each frame
                    
            ################################################################
            #TORCH OBJECT DETECTION
            ################################################################

            # Get dimensions of the current video frame
            x_shape = image.shape[1]
            y_shape = image.shape[0]

            # Apply the Torch YoloV5 model to this frame
            results = model(image)
            
            # Extract the labels and coordinates of the bounding boxes
            labels, cords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

            numberOfLabels = len(labels)
            
            # declare empty array of objects found
            objectsFound = []

            for i in range(numberOfLabels):
                row = cords[i]
                # Get the class number of current label
                class_number = int(labels[i])
                # Index colors list with current label number
                color = COLORS[class_ids[class_number]]
                
                # If confidence level is greater than 0.2
                if row[4] >= 0.4:
                    # Get label to send to dashbaord
                    label = classes[class_number]
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                    # append coords and label so it can be analyzed
                    objectsFound.append([x1, y1, x2, y2, label])

                    # If global enable flag is set true then show boxes
                    # Draw bounding box
                    cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                    # Give bounding box a text label
                    cv2.putText(image, str(classes[int(labels[i])]), (int(x1)-10, int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)       

        # will need to instead output to dashboard.html
        '''
        if frame is not None:
            # Display the resulting frame
            cv2.imshow('frame', image)

            # Press q to close the video windows before it ends if you want
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break
        else:
            print("Frame is None")
            break
        

    # When everything done, release the capture
    vcap.release()
    cv2.destroyAllWindows()
    print("Video stop")
    '''

if __name__ == '__main__':
    url = hls_stream()
    runModels(url)
