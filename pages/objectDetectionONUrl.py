import boto3
import cv2
import os
import torch
import numpy as np

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

# run yolo based on video stream from kinesis
def runYolo(url):
    vcap = cv2.VideoCapture(url)

    # Torch code adapted from https://github.com/akash-agni/Real-Time-Object-Detection/blob/main/Object_Detection_Youtube.py
    model_weight_path = os.path.join(os.getcwd(), 'PatrolBot/model_weights/best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', model_weight_path)
    
    # Extract the names of the classes for trained the YoloV5 model
    classes = model.names
    class_ids = [0,1,2,3]
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    while(True):
        # Capture frame-by-frame
        ret, frame = vcap.read()

        if ret:

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip video frame, so it isn't reversed
            image = cv2.flip(image, 1)

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
    runYolo(url)