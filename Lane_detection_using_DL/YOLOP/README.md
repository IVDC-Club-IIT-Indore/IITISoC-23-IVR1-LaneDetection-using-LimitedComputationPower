# YOLOP
Earlier we used traditional methods(canny edge and Hough Transform) for the lane detections but it was not a good option for complex curves , shadows, varying colors on road, bad weather conditions. Then we extended our approach to use Deep Learning models and tried different CNN models for single Lane detection,, we got satisfying results but due to limitations like multiple lanes were not detected, occlusionsdue to vehicles and pedestrians were affecting lane detection so  we extended our approach to models which can give good results for multiple lane detection and use the global context information to infer the vacancy or occluded part . By using YOLOP we tried **multiple lane detection, drivable area ** although **object detection** is not part of our project . Constraint of computational power and deployment of model on edge devices was kept in mind. This model can reach real-time on embedded device Jetson TX2 with TensorRT deployment

## Architecture


![yolop_architecture](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/assets/117708050/7e55492f-ff8d-4f16-a8a6-e73a23c786d6)

A lightweight CNN is used as the encoder to extract features from the image. Then these feature maps are fed to three decoders to complete their respective tasks.

## Results
Output videos on which model is tested can be seen by [clicking here](https://drive.google.com/drive/u/0/folders/1_CCMGf2t9jftklbTtzbtalFi4jYPxBXe)
<br>



## Directory structure
```bash
YOLOP
├─inference
│ ├─images   # inference images
│ ├─output   # inference result
├─lib
│ ├─config/default   # configuration of training and validation
│ ├─core    
│ │ ├─activations.py   # activation function
│ │ ├─evaluate.py   # calculation of metric
│ │ ├─function.py   # training and validation of model
│ │ ├─general.py   #calculation of metric、nms、conversion of data-format、visualization
│ │ ├─loss.py   # loss function
│ │ ├─postprocess.py   # postprocess(refine da-seg and ll-seg, unrelated to paper)
│ ├─dataset
│ │ ├─AutoDriveDataset.py   # Superclass dataset，general function
│ │ ├─bdd.py   # Subclass dataset，specific function
│ │ ├─hust.py   # Subclass dataset(Campus scene, unrelated to paper)
│ │ ├─convect.py 
│ │ ├─DemoDataset.py   # demo dataset(image, video and stream)
│ ├─models
│ │ ├─YOLOP.py    # Setup and Configuration of model
│ │ ├─light.py    # Model lightweight（unrelated to paper, zwt)
│ │ ├─commom.py   # calculation module
│ ├─utils
│ │ ├─augmentations.py    # data augumentation
│ │ ├─autoanchor.py   # auto anchor(k-means)
│ │ ├─split_dataset.py  # (Campus scene, unrelated to paper)
│ │ ├─utils.py  # logging、device_select、time_measure、optimizer_select、model_save&initialize 、Distributed training
│ ├─run
│ │ ├─dataset/training time  # Visualization, logging and model_save
├─tools
│ │ ├─demo.py    # demo(folder、camera)
│ │ ├─test.py    
│ │ ├─train.py
├─weights    # Pretraining model  

    

```

### Requirements :
 Give a command as :
 ```
cd YOLOP
pip install -r requirements.txt

 ```

### Download :
Download the weight from [here](https://drive.google.com/drive/u/0/folders/1_CCMGf2t9jftklbTtzbtalFi4jYPxBXe) and save in ```weights``` folder .


## Key files
#### Training:
[train.py](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/YOLOP/tools/train.py) :
If you want to train the model for single task please modify the corresponding configuration in ```./lib/config/default.py```to ```True```(which is ```False``` by default). Training configuration can be set in ```./lib/config/default.py```

```
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task
```
###### Start Training
```
python tools/train.py
```

#### Evaluation
[test.py](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/YOLOP/tools/test.py) : Evaluation configuration cabe set in ``` .lib/config/default.py ```
Give the following command to evaluate :
```
python tools/test.py --weights End-to-End.pth  # create a folder named weights and put downloaded weight file in that folder
```

#### Visulaise
[demo.py](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/YOLOP/tools/demo.py) :
If you want to test on images , save your images in ```inference/images``` and select your save directory (you can make a seperate folder for output images for now let it be ```output```)
Give following command  to test in images
```
python tools/demo.py --source inference/images --save-dir inference/output
```
To test on video store your video in ``` --source ``` (make a folder for videos) and select you save directory
```
python tools/demo.py --source inference/videos --save-dir inference/output
```
To test on web-cam
```
python tools/demo.py --source 0 -- save-dir inference/output
```
## Refrences
- article : https://link.springer.com/article/10.1007/s11633-022-1339-y
  
