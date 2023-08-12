import codecs
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from imagezmq import ImageHub

import time

# Load the pre-trained model with custom loss functions...
model = load_model(r"C:\Users\arjun\Downloads\LLDNet.h5")

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def find_avg_angle(far_center_x , near_center_x , near_center_y ,far_center_y ):
    

    sum_angles = 0 
    
    #centers = [ i for i in range(-5, 6,1) ]
    #centers = [  i for i in range(-20, 21,1) ]
    centers = [ i for i in range(-10, 11,1) ]
    
    for i in centers:
        
        rad_angle = np.arctan(  ( far_center_x - (near_center_x + i)) / (near_center_y - far_center_y ) )
        angle = rad_angle*(180/np.pi)
        sum_angles += angle
        print(angle)
    
    return sum_angles/len(centers)


# Initialize ImageHub to receive frames from the webcam on the Raspberry Pi
print('before imagehub')
image_hub = ImageHub(open_port='tcp://*:5555', REQ_REP=True)
print('after imagehub')

while True:  # Show streamed images until Ctrl-C
    rpi_name, image = image_hub.recv_image()
    #image_hub.send_reply(b'hii')
    print('change')
    lanes = Lanes()
    
    small_img = cv2.resize(image, (160, 80))
    small_img_preprocessed = small_img  # Replace with your preprocessing logic
    
    
    prediction_matrix = model.predict(np.expand_dims(small_img_preprocessed, axis=0))[0] * 255.0  #1 channeled image
    lanes.recent_fit.append(prediction_matrix)
    print(type(prediction_matrix))
    
    print(len(lanes.recent_fit))
    #cv2.imshow('recentfit', lanes.recent_fit() )
    if len(lanes.recent_fit) > 5:
       lanes.recent_fit = lanes.recent_fit[1:]
       
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)
    
    
    print(len(lanes.avg_fit.shape))
    #cv2.imshow('avgfit', lanes.avg_fit() )
    
    
# Calculate the center of the drivable area near the vehicle
    print("prediction shape : ", prediction_matrix.shape )
    #print(prediction_matrix)
    cv2.imshow('prediction', prediction_matrix)
    near_center_x = len(prediction_matrix[0]) // 2
    image_height = len(prediction_matrix)
    
    near_center_y = image_height
    print('near center: ' , near_center_x ,image_height )
   
# Calculate the center of the detected lane image farther down the road
# Find the indices where the values are above a certain threshold
    threshold = 0.5  # You can adjust this threshold based on your specific case
    
    
    lane_indices = np.where(prediction_matrix > threshold)[0]  #improve to get actual centre
    
    print('lane indices: ' , len(lane_indices))
    far_center_y = int(np.mean(lane_indices)) if len(lane_indices) > 0 else near_center_x #avg row 
    
    
    out =  np.where( prediction_matrix[far_center_y] > threshold)[0]
    
    far_center_x = int(np.mean( out  ))
    
    print('far center : ', far_center_x, far_center_y)
# Calculate the angle based on the difference between these centers
    image_width = len(prediction_matrix[0])
    #max_angle = 45.0  # Maximum angle in degrees
    
    
    angle = find_avg_angle(far_center_x , near_center_x , near_center_y ,far_center_y )
    #rad_angle = np.arctan(  ( far_center_x - near_center_x) / (near_center_y - far_center_y ) )
    #angle = rad_angle*(180/np.pi)
    #lane_angle = ((far_center - near_center) / image_width ) * max_angle
    
    strangle = str(angle)
    b_string = codecs.encode(strangle, 'utf-8')
    
    image_hub.send_reply(b_string)
    
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
    lane_image = cv2.resize(lane_drawn, (640, 480))
    
    image = cv2.resize(image, (640, 480))
    lane_image = lane_image.astype(np.uint8)
    
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    cv2.imshow('Lane Detection', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

image_hub.close()
cv2.destroyAllWindows()
