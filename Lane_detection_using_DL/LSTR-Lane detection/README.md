## LSTR-Lane Detection
Initially we tried conventional methods including Canny edge and Hough transform for Lane detection. But this was not robust enough. Then we extended our approach 
to deep learning models and worked on different CNN models. This yeilded good results, but we wanted to improve on this too by incorporating **drivable area 
segmenation** and **lane detection** both.(Although we avoided Obstacle and object detection in the scope of our project).Constraint of computational power and 
deployment of model on edge devices was kept in mind. This model can reach real-time on embedded device Jetson TX2 with TensorRT deployment
