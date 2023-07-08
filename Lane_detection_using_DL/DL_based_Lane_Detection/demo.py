import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from tensorflow import keras
import time
import matplotlib.image as mpimg

model = keras.models.load_model('full_CNN_model.h5')

class Lanes():
  def __init__(self):
    self.recent_fit = []
    self.avg_fit = []


video_capture = cv2.VideoCapture('harder_challenge_video.mp4')
video_capture.set(3,640)
video_capture.set(4,480)

fps =video_capture.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))
time_taken=[]

while(video_capture.isOpened()):
  lanes = Lanes()


  # Capture the frames
  ret,image = video_capture.read()
  if not ret:
       print("Can't receive frame (stream end?). Exiting ...")
       break
   
  if ret: 

      small_img = cv2.resize(image, (160,80))
      small_img = np.array(small_img)
      small_img = small_img[None,:,:,:]
      start_time = time.time()


      prediction = model.predict(small_img)[0] * 255
      total_time = time.time() - start_time

      fps = int(1/total_time)
      print(fps)

      lanes.recent_fit.append(prediction)

      if len(lanes.recent_fit) > 5:
          lanes.recent_fit = lanes.recent_fit[1:]

      lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

      blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
      lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

      lane_image = cv2.resize(lane_drawn, (640,480))
      image = cv2.resize(image, (640,480))

      lane_image = lane_image.astype(np.uint8)
      result = cv2.addWeighted(image, 1, lane_image, 1, 0)

      out.write(result)
      

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()


