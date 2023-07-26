## Hybridnets
<hr>
<p>
Earlier we used traditional methods(canny edge and Hough Transform) for the lane detections but it was not a good option for complex curves , shadows, varying colors on road, bad weather conditions.
Then we extended our approach to use Deep Learning models and tried different CNN models for single Lane detection,, we got satisfying results but due to limitations like multiple lanes were not detected, occlusions
due to vehicles and pedestrians were affecting lane detection so we extended our approach to models which can give good results for multiple lane detection and use the global context information to infer the vacancy or 
occluded part . Then we tried various models among which Hybridnets is one of the model .
</p>
<br>
HybridNets is an end2end perception network for multi-tasks like  traffic object detection, drivable area segmentation and lane detection. HybridNets can run real-time on embedded systems, and obtains SOTA Object Detection, Lane Detection on BDD100K Dataset.

## Architecture :
<hr>

![Screenshot 2023-07-26 092336](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/assets/117708050/92699498-db25-438a-a532-1a8f8e17f2bf)


<br>
HybridNets Architecture has one encoder: backbone network and neck network; two decoders: Detection Head and Segmentation Head
<br>

#### Directory structure :
```
Hybridnets
├─hybridnets
| ├─hybridnets.py
| ├─__init__.py
| ├─remove_unused_nodes.py
| ├─utils.py
├─models
| ├─anchors_384x512.npy
| ├─hybridnets_384x512.onnx
├─Test_videos      # add you videos on which you want to detect lanes
├─output_videos     # give path of this folder whi;e saving the results
├─video_road_detection.py
```
#### Requirements :

Give a command :
```
cd Hybridnets
pip install -r requirements.txt

```
#### Downloads :

<br>
To download weights 

[click here](https://drive.google.com/drive/folders/1u-gjL8CBy_7SahyeMcHbwK1FLJKIiehT?usp=drive_link)

and save in models folder.

If you want to download original pytorch model [click here](https://github.com/datvuthanh/HybridNets)

#### Key File :

 [video_road_detection.py](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/Hybridnets/video_road_detection.py)  - 
 Run this file for detecting lanes on video saved  in ```Test_videos``` folder

#### Example :

For more examples [click here](https://drive.google.com/drive/folders/1quIkeXg076_x3DBXYCNrjGL8Qj7MdCqO?usp=drive_link)

#### Refrences :

- HybridNets model : https://github.com/datvuthanh/HybridNets
- Original Paper : https://arxiv.org/abs/2203.09035
 
