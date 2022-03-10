# Create a list of frames and run the custom action detection model on them

import matplotlib.pyplot as plt
import numpy as np

from mxnet import gluon, nd, image

from gluoncv.model_zoo import get_model

import cv2

import torchvision.transforms as T

from gluoncv.utils.filesystem import try_import_decord

decord = try_import_decord()

# get list of frame numbers for fast portion of neural network
fast_frame_id_list = range(0, 64, 2)
# get list of frame numbers for slow portion of neural network
slow_frame_id_list = range(0, 64, 16)
# combine the two lists
frame_id_list = list(fast_frame_id_list) + list(slow_frame_id_list)

Capture = cv2.VideoCapture(0)

# frames captured initially set to 0
frameCounter = 0
# declare a list of frames
clip_input = []
# declare a list of slow frames
slow_input = []

while frameCounter <=70:
    ret, frame = Capture.read()

    if ret:
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

print('Video data is preprocessed.')

model_name = 'i3d_resnet50_v1_custom'
net = get_model(model_name, nclass=2, pretrained=False)
# put your path to your weights here
net.load_parameters('/Users/brandonbanuelos/Downloads/ActionDetection.params')
print('%s model is successfully loaded.' % model_name)

pred = net(nd.array(clip_input))

classes = ['normal', 'abnormal']
topK = 2
ind = nd.topk(pred, k=topK)[0].astype('int')
print('The input video clip is classified to be')
predictions = []

for i in range(topK):
    predictions.append([classes[ind[i].asscalar()],nd.softmax(pred)[0][ind[i]].asscalar()]) 

# extract the action with the highest confidence level
bestPrediction = predictions[0]
bestAction = bestPrediction[0]
bestConfidence = bestPrediction[1]

# print out best action for stats
print(bestAction, " with confidence ", bestConfidence)