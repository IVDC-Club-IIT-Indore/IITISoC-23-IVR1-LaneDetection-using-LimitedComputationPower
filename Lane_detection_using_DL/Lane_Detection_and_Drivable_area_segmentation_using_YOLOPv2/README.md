## YOLOPv2
YOLOPv2 is an excellent multi-task network based on YOLOP. By using YOLOPv2 we tried **multiple lane detection** and **drivable area segmentation** and **object detection**(although **object detection** is not part of our project). Constraint of computational power and deployment of model on edge devices was kept in mind. This model can reach real-time on embedded device Jetson TX2 with TensorRT deployment.

## Models
The pretrained model file can be downloaded from [here](https://drive.google.com/drive/folders/16OZK_zvGecemXcUylyX_sc_z8lL8PS24?usp=sharing).

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
