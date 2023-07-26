## YOLOPv2
YOLOPv2 is an excellent multi-task network based on YOLOP. By using YOLOPv2 we tried **multiple lane detection** and **drivable area segmentation** and **object detection**(although **object detection** is not part of our project). Constraint of computational power and deployment of model on edge devices was kept in mind. This model can reach real-time on embedded device Jetson TX2 with TensorRT deployment.

## Requirements
Check the requirements.txt file. Additionally you'll have to download the pretrained Pytorch model and also download and extract BDD100K train, val and test with annotations.

## Dataset
The dataset used is BDD100K.

## Data Preperation
Download Images from [images](https://bdd-data.berkeley.edu/).

Download the annotations of drivable area segmentation from [segments](https://drive.google.com/file/d/1xy_DhUZRHR8yrZG3OwTQAHhYTnXn7URv/view).

Download the annotations of lane line segmentation from [lane](https://drive.google.com/file/d/1lDNTPIQj_YLNZVkksKM25CvCHuquJ8AP/view).

Set the directory structure like:

```
/data
    bdd100k
        images
            train/
            val/
            test/
        segments
            train/
            val/
        lane
            train/
            val/
```

## Installation 
Create a folder namde YOLOPv2 in your device and navigate to there.
For example:

```
cd C:\
mkdir YOLOPv2
cd C:\LSTR
```

then create a conda environment and install the requirements using pip command
```
conda create --name YOLOPv2
conda activate YOLOPv2
pip install -r requirements.txt
```
afterwards you'll have to download the pretrained model file from [model](https://drive.google.com/drive/folders/16OZK_zvGecemXcUylyX_sc_z8lL8PS24) and upload it in weights folder inside data. The directory structure should look like this:
```
/data
    weights
        model_url.txt
        yolopv2.pt
    demo
           
```

## Day-time and Night-time visualization results
<img src="https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/Lane_Detection_and_Drivable_area_segmentation_using_YOLOPv2/data/demo/day_and_night_results.jpg" width="400" height="400" />


## Models
The pretrained model file(.pt) can be downloaded from [here](https://drive.google.com/drive/folders/16OZK_zvGecemXcUylyX_sc_z8lL8PS24?usp=sharing). In addition you can also download an ONNX model file.

## Results
Lanes predicted using this model along with the input videos and images can be found [here](https://drive.google.com/drive/folders/1lvab6Yn6sYm76Rg7pzsxMBhDTrjDQ41m?usp=sharing).

## Examples
**Image Inference :**
After navigating to the directory in which demo.py file is present, the image inference can be done by replacing input_image.jpg with the suitable image path in your device
```
!python demo.py  --source data/input_image.jpg
```

**Video Inference :**
Similarly replace the input_video.mp4 with desired video path in your device.
```
!python demo.py  --source data/input_video.mp4
```

## Key Files

1.[utils](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/Lane_Detection_and_Drivable_area_segmentation_using_YOLOPv2/utils/utils.py): Contains the necessary classes and functions needed to run the video and image tests.

2.[demo](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/Lane_Detection_and_Drivable_area_segmentation_using_YOLOPv2/demo.py): Pipeline return output images and videos with lanes predicted using the already trained model and input feed.

3.[custom_demo](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/Lane_Detection_and_Drivable_area_segmentation_using_YOLOPv2/custom_demo.py): A slightly modified version for which the code is present in the custom_demo.py file. It is an almost exact copy of the original with a few minor changes. Basically it detects only the lanes and no drivable area is detected.

References :

[1]CAIC-AD / YOLOPv2(2022), Github Repository, https://github.com/CAIC-AD/YOLOPv2

[2]YOLOPv2 NCNN C++ Demo: [YOLOPv2-ncnn](https://github.com/FeiGeChuanShu/YOLOPv2-ncnn).

[3]YOLOPv2 ONNX and OpenCV DNN Demo: [yolopv2-opencv-onnxrun-cpp-py](https://github.com/hpc203/yolopv2-opencv-onnxrun-cpp-py).
