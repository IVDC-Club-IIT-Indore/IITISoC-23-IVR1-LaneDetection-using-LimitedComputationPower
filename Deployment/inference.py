import cv2
import numpy as np
from tensorflow.keras.models import load_model
from imagezmq import ImageHub

import time

# Load the pre-trained model with custom loss functions...
model = load_model(r"C:\Users\arjun\OneDrive\Desktop\IITISoc_direct_Implementation\CNN1(Ampady)\Trained_model.h5")

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

# Initialize ImageHub to receive frames from the webcam on the Raspberry Pi
print('before imagehub')
image_hub = ImageHub(open_port='tcp://*:5555', REQ_REP=True)
print('after imagehub')

while True:  # Show streamed images until Ctrl-C
    rpi_name, image = image_hub.recv_image()
    image_hub.send_reply(b'OK')
    
    lanes = Lanes()
    
    small_img = cv2.resize(image, (160, 80))
    small_img_preprocessed = small_img  # Replace with your preprocessing logic
    
    prediction = model.predict(np.expand_dims(small_img_preprocessed, axis=0))[0] * 255.0
    lanes.recent_fit.append(prediction)
    
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]
        
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)
    
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
