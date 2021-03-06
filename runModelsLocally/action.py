# -*- coding: utf-8 -*-
"""Models on Stream (No Yolo).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gcJ_lc_i6bO1VJXIGv4ZxpVR33J1bdWq
"""

#!pip3 install mxnet
#!pip3 install gluoncv
#!pip3 install torch
#!pip3 install torchvision
#!pip3 install opencv-python
#!pip3 install numpy
#!pip3 install PyYAML
#!pip3 list

# import items for object detection
import cv2
import os

#from google.colab.patches import cv2_imshow
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

# run yolo based on video stream from kinesis
def runModels(url):
    vcap = cv2.VideoCapture(url)
    vcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

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
              print(bestAction, " with confidence ", bestConfidence)

              # reset frame counter to 0
              frameCounter = 0
              # empty the frames lists so they can be collected again
              # declare a list of frames
              clip_input = []
              # declare a list of slow frames
              slow_input = []
          
      # will need to instead output to dashboard.html
      
      if frame is not None:
          # Display the resulting frame
          cv2_imshow(image)

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
    

if __name__ == '__main__':
    url = 'https://cph-p2p-msl.akamaized.net/hls/live/2000341/test/master.m3u8'
    # run models on URL and show output 
    runModels(url)