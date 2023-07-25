## LSTR-Lane Detection
Initially we tried conventional methods including Canny edge and Hough transform for Lane detection. But this was not robust enough. Then we extended our approach 
to deep learning models and worked on different CNN models. This yeilded good results, but we wanted to improve on this too by incorporating **drivable area 
segmenation** and **lane detection** both.(Although we avoided Obstacle and object detection in the scope of our project).Constraint of computational power and 
deployment of model on edge devices was kept in mind. This model can reach real-time on embedded device Jetson TX2 with TensorRT deployment.

## Requirements
Check the requirements.txt file. Additionally you'll have to download the onnx models, pretrained Pytorch model and also download and extract TuSimple train, val and test with annotations.

## Data Preperation
Download and extract TuSimple train, val and test with annotations from [TuSimple](https://github.com/TuSimple/tusimple-benchmark). Set the directory structure like:
```
TuSimple/
    LaneDetection/
        clips/
        label_data_0313.json
        label_data_0531.json
        label_data_0601.json
        test_label.json
    LSTR/
```

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

## Training
To train a model:
(if you only want to use the train set, please see ./config/LSTR.json and set "train_split": "train")
```
python train.py LSTR
```

## Examples
**Image Inference :**
```
python image_lane_detection.py
```
**Video Inference:**
```
python video_lane_detection.py
```
## Key Files: 
1. [image_lane_detection](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/LSTR-Lane%20detection/image_lane_detection.py) and [video_lane_detection](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/LSTR-Lane%20detection/video_lane_detection.py): These two pipelines return output images and videos with lanes predicted using the already trained model and input feed.

2.[train](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/LSTR-Lane%20detection/train.py): After downloading the training images and labels above, this is the training file to obtain the trained file for inference.

3.[lstr](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/LSTR-Lane%20detection/lstr/lstr.py): Contains the necessary classes and functions needed to run the video and image tests.

References:
[1] ibaiGorordo / ONNX-LSTR-Lane-Detection(2021), Github repository, https://github.com/ibaiGorordo/ONNX-LSTR-Lane-Detection

[2]liuruijin17 / LSTR(2020), Github Repository, https://github.com/liuruijin17/LSTR

[3]Vision Transformers for Computer Vision[+],https://towardsdatascience.com/using-transformers-for-computer-vision-6f764c5a078b
