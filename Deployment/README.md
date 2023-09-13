## Deployment
Before deploying our models onto edge computing device like te NVIDIA Jetson Xavier NX  , we first thought of deploying it on an Raspberry pi 4 Model B (8 GB RAM).

# Vehicle Development
The vehicle selection, design and testing were done in many steps.Some of the major improvements include:

1. Initial Test Model 

<img src="https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Deployment/data/prototype_1.jpg" width="400" height="400" />

Tough this model failed, a lot of meaningful insights were gained after extensive troubleshooting. Some of them include :
- The camera needs to be mounted high above the vehicle for better detection.
- The weight balancing needs to be done properly

2. Second Test Model

<img src="https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Deployment/data/prototype_2.jpg" width="400" height="400" />

Tough this model resolved the issue of camera mount, the weight balance was still an issue. Additionally the traction and load capacity of the wheels with further weight increase was also an issue. Imbalance resulted in rotation difference on sides which led to undesirable trajectories.

3. Third Test Model

<img src="https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Deployment/data/prototype_3.jpg" width="400" height="400" /> 

The chasis as a whole was changed to metallic 2 level chasis and the motors were upgraded to higher power 300rpm, 12V motors. Significant modelling change was made here and all the issues were resolved in this model. The skid steer based vehicle was ready completely. But here we realized that at simple curves, at high speeds skid steer vehicle model returned commands with high errors that couldn't be even tuned using PID Tuning. And there was failure in case of complex curves.

4.FINAL VEHICLE:

Finally learning from all the 3 test models, and after lots of troubeshooting we decided to buy a steering arm mechanism based RC Car with metal shock absorbers, whole metallic bearings, pneumatic tires, high speed carbon brush motor, metal drive bone, metallic gears,etc. This RC Car was then made completely autonomous. The radio control for the High speed carbon brush motor was retained as such(linear motion), but the servo commands were given to the Model which was run on an Rapberry Pi. The model frame processing time rate was tuned close to the servo dutyCycle. Thus the Vehicle would autonomously be able to navigate through the track, meanwhile also incorporating Obstacle Detection algorithm.

<img src="https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Deployment/data/final_vehicle.jpg" width="400" height="400" /> 

TRACK: 

<img src="https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Deployment/data/track.jpg" width="400" height="400" /> 
