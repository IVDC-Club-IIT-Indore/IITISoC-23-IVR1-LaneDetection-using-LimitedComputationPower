import cv2

import time 
from hybridnets import HybridNets, optimized_model

import torch
from torchsummary import summary
# Initialize video
# cap = cv2.VideoCapture("test.mp4")

cap = cv2.VideoCapture(r"C:\Users\91701\Git-Repositories\ONNX-HybridNets-Multitask-Road-Detection\Test_videos\challenge1.mp4")
start_time = 0 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time)
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize road detector
model_path = (r"C:\Users\91701\Git-Repositories\ONNX-HybridNets-Multitask-Road-Detection\models\hybridnets_384x512.onnx")
anchor_path = (r"C:\Users\91701\Git-Repositories\ONNX-HybridNets-Multitask-Road-Detection\models\anchors_384x512.npy")
optimized_model(model_path) # Remove unused nodes
roadEstimator = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)


model = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)
dummy_input = torch.randn(1, 3, 480, 1280)  # Assuming input_height and input_width are the dimensions of your input

# Pass the model and the dummy input to the summary function
summary(model, input_size=(3, 480, 1280))


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
cv2.namedWindow("Road Detections", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break
    
    try:
    		# Read frame from the video
        ret, new_frame = cap.read()
    		
            
    			
        if not ret:	
            break
    
    except:
        continue
    	

    height,width = new_frame.shape[:2]
    print("Frame dimensions: {} x {}".format(width, height))	
    start_time = time.time()
    seg_map,_ , _ = roadEstimator(new_frame)
    combined_img = roadEstimator.draw_segmentation(new_frame)
    
    total_time = time.time() - start_time
    
    fps = 1/(time.time() - start_time)
    fps = int(fps)
    print(fps)
    
    cv2.imshow("Road Detections", combined_img)
    out.write(combined_img)

out.release()