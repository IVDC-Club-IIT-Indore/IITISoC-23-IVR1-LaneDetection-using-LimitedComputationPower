## LSTR-Lane Detection
Initially we tried conventional methods including Canny edge and Hough transform for Lane detection. But this was not robust enough. Then we extended our approach 
to deep learning models and worked on different CNN models. This yeilded good results, but we wanted to improve on this too by incorporating **drivable area 
segmenation** and **lane detection** both.(Although we avoided Obstacle and object detection in the scope of our project).Constraint of computational power and 
deployment of model on edge devices was kept in mind. This model can reach real-time on embedded device Jetson TX2 with TensorRT deployment.

## Requirements
Check the requirements.txt file. Additionally you'll have to download the onnx models and the pretrained Pytorch model

## Installation
Navigate to C drive and create a folder named LSTR 
```
cd C:\
mkdir LSTR
cd C:\LSTR
```
create a conda environment. Then install the requirements using pip command 
```
conda create --name LSTR
conda activate LSTR
pip install -r requirements.txt
```

## ONNX model 
The model zip file can be found [here](https://drive.google.com/drive/folders/1oDUhy5k3RyvNLO8nYoVOLng55ua8-z2q?usp=sharing).

## Orginal Pytorch model
The pretrained pytorch model can be found [here](https://drive.google.com/drive/folders/1zMSSeZdBQ1s7taKrhU-mtEiTnz5goFTK?usp=sharing).

## Results
Lanes predicted using this model along with the input videos and images can be found [here](https://drive.google.com/drive/folders/1O5_s5Do6JK9OnM6kb-BMITNsS5R5h06w?usp=sharing).

## Key Files

