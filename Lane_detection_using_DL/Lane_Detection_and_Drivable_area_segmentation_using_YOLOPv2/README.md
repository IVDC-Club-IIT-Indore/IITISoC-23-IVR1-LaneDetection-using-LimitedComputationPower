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

**References :**

[1]CAIC-AD / YOLOPv2(2022), Github Repository, https://github.com/CAIC-AD/YOLOPv2

[2]YOLOPv2 NCNN C++ Demo: [YOLOPv2-ncnn](https://github.com/FeiGeChuanShu/YOLOPv2-ncnn).

[3]YOLOPv2 ONNX and OpenCV DNN Demo: [yolopv2-opencv-onnxrun-cpp-py](https://github.com/hpc203/yolopv2-opencv-onnxrun-cpp-py).
