import socket

import cv2
import imagezmq
 
sender = imagezmq.ImageSender(connect_to='tcp://10.202.1.197:5555')  
rpi_name = socket.gethostname() # send RPi hostname with each image
video_capture = cv2.VideoCapture(0)
 
 
 # time.sleep(0.5)  # allow camera sensor to warm up
while True:  # send images as stream until Ctrl-C
	ret,image = video_capture.read()
	if ret:
		out = sender.send_image(rpi_name,image)
		print(out)
	else:
		print('Error in capturing frame')
