# IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower

<img src="https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/assets/125551038/10cf6b34-fb81-4bc9-90c3-4c2df9ebd510" width = 848 height = 480>


## Goal :
To Develop a robust lane detection pipeline that consumes meager computational resources (No GPU allowed, limited CPU and RAM usage) and could be deployed on NVIDIA Jetson Nano Board or even a Raspberry Pi board

People Involved :


Mentors:
- [Kshitij M. Bhat](https://github.com/KshitijBhat) 
- [Sairaj R. Loke](https://github.com/SairajLoke)

Members:
<br>
- [Bhawna Chaudhary](https://github.com/WebWizard104)
- [Arjun S Nair](https://github.com/arjun-593)
- [AMPADY B R](https://github.com/ampady06)
- [Aditya Singh](https://github.com/AdityaSingh1199)

### Outline :
This repository contains the implementation of a lane detection system using two different approaches. The main goal of this project was to understand the core fundamentals of lane detection in images. The two approaches utilized are as follows:
<br>
- [Approach-1](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/NonDL_Lane_Detection.ipynb) - Foundation Approach( Lane Detection using Canny Egde and Hough transform)
- [Approach-2](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/tree/main/Lane_detection_using_DL) - Advanced approach with Deep Learning.

### Approach-1 :
In this approach, traditional computer vision techniques were employed to detect lanes in an image. The input pipeline for Approach 1 consists of a sequence of techniques applied to the input image to detect lane lines. Each step in the pipeline is essential for accurate and reliable lane detection. Sequence is as follows:

- **Preprocessing** : The input image was preprocessed to enhance lane features and reduce noise.
- **Canny Edge Detection** : The Canny edge detection algorithm was applied to extract edges in the image.
- **Region of Interest selection** : When finding lane lines, we don’t need to check the cloud and the
                                     mountains in a image. Thus the objective of this technique is to concentrate on the 
                                     region of interest for us that is the road and lane on the road

- **Hough Transform** : The Probabilistic Hough transform algorithm was used to detect lines in the edge-detected image, 
                        which represent potential lane markings.
- **Post-processing** : The detected lines were further processed to combine and extend them to form complete lane 
                        boundaries.
#### Predicted Results :
<img src="https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/data/demo/predicted_result.jpg" width="400" height="400" />



#### Disadvantages :
- Fails in complex curved roads.
- Does not give satisfactory results in rainy and foggy environment.


### Approach-2 :
In this advanced approach, we have explored the effectiveness of deep learning models, including Convolutional Neural Networks (CNNs) and Transformer models, for accurate and efficient lane detection. While CNNs are widely known for their image analysis capabilities, we also investigated the potential of Transformer-based models, such as LSTR (Lane Shape Transformer), which are specifically designed for sequence-to-sequence tasks like lane detection.

#### Model Selection : 
We carefully curated and tested several state-of-the-art deep learning models for lane detection. The following models were among those evaluated :
<br>
<ul>
<li> <strong>3 CNN Models </strong> : We explored three different CNN architectures that we found through YouTube and GitHub. These models were chosen for their effectiveness in image analysis tasks and had demonstrated promising results in lane detection scenarios. </li>
<br>
  
<li> <strong>YOLOP and YOLOPv2 </strong> : We experimented with YOLOP and its upgraded version, YOLOPv2, which are well-known for their real-time object detection capabilities. We adapted these models for lane detection and evaluated their performance.</li>
<br>
<li> <strong> HybridNets </strong> : HybridNets is a popular deep learning architecture specifically designed for lane detection. We examined its performance and capabilities for detecting complex lane geometries.</li>
<br>
<li> <strong>LSTR(Lane Shape Transformer) </strong> : While the Transformer-based model LSTR showed impressive frames per second (fps) performance, we found that its detection results did not meet our expectation.In this Transformer-based lane detection architecture, the model consists of several key components:
  <hr>
  <ul>
   <li> Backbone: The backbone extracts low-resolution features from the input image I and converts them into a sequence S by collapsing the spatial dimensions. 
   </li>

  <li> Reduced Transformer Network: The sequence S, along with positional embeddings Ep, is fed into the transformer encoder to produce a representation sequence Se. The transformer is responsible for capturing long- 
    range  dependencies and interactions within the sequence.</li>

  <li> Decoder: The decoder generates an output sequence Sd by attending to an initial query sequence Sq and a learned positional embedding ELL, which implicitly learns positional differences. The decoder computes 
    interactions with Se and Ep to attend to related features. </li>

  <li> Feed-forward Networks (FFNs): Several feed-forward networks are employed to directly predict the parameters of proposed lane outputs. </li>

  <li> Hungarian Loss: The model utilizes the Hungarian Loss, a specific loss function tailored for lane detection tasks, to optimize the parameters and ensure accurate lane predictions. </li>

 <li> The architecture leverages the power of the transformer model for sequence-to-sequence tasks, allowing for more effective lane detection, especially in scenarios involving curved lanes and complex lane geometries.
 </li>
 </ul>
 </li>
  </ul>

  #### Predicted Results :
  the result for the three CNN models, YOLOP, YOLOPv2, LSTR and hybridnets can be found [here](https://drive.google.com/drive/folders/116NJ985ODlUZG95Dph85_vxKQ6EUXhgI?usp=drive_link)

## System Specifications :
All the work is done in 3 devices(2 of which are same) namely asus vivobook 15 pro and HP Pavilion Gaming 15 ec2008AX. 
**HP Pavilion Gaming 15 ec2008AX** has **Processor AMD Hexa Core Ryzen 5 5600H**, **RAM 8 GB DDR4 RAM** meanwhile the vivobook 15 pro is a **12th Gen Intel Core H-series processors with 16 GB of LPDDR5 RAM, and an NVIDIA ® GeForce ® RTX ™ 3050 Ti GPU**


#### Model Evaluation :
During our comprehensive testing, we considered multiple deep learning architectures, such as CNNs, HybridNets, YOLOP, YOLOPv2, and LSTR. Each model underwent rigorous evaluation using performance metrics like Mean Average Precision (mAP), Intersection over Union (IoU), and inference speed (fps), precision, recall, f1 score.
<br>
**Comparison of 3 CNN Models**
<table align = "center">
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Size(KB)</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>FPS</th>
      <th>Dice coefficient</th>
      <th>IoU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CNN 2</td>
      <td> 129,498 </td>
      <td> 580.15 </td>
      <td> 0.939 </td>
      <td> 0.747 </td>
      <td> 0.8327 </td>
      <td> 12 </td>
      <td> 0.8327 </td>
      <td> 0.72</td>
    </tr>
    <tr>
      <td> CNN 3  </td>
      <td> 181,693 </td>
      <td> 150.02 </td>
      <td> 0.980 </td>
      <td> 0.731  </td>
      <td> 0.837 </td>
      <td> 15 </td>
      <td> 0.837 </td>
      <td> 0.72 </td>
    </tr>
    <tr>
      <td> CNN 1 </td>
      <td> 125,947 </td>
      <td> 55 </td>
      <td> 0.97 </td>
      <td> 0.984 </td>
      <td> 0.99 </td>
      <td> 5 </td>
      <td> 0.987 </td>
      <td> 0.976 </td>
    </tr>
  </tbody>
</table>
<br>

<strong> Graphical representation of comparison of models </strong>

![Untitled design](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/assets/125551038/bf0d2410-f003-48e4-af58-7d7d436c8dc3)

<img src="https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/assets/125551038/91740d33-e381-499f-ac3a-54c6623b9273" width = 1280 height = 180 >


<strong> Comparison of YOLOP, YOLOPV2, Hybridnets : </strong>
<table align="center">
  <thead>
    <tr>
      <td> <strong> Model </strong> </td>
      <td> <strong> Parameters(million) </strong> </td>
      <td> <strong> Size(KB) </strong> </td>
      <td> <strong> Accuracy </strong></td>
      <td> <strong> IoU(Lane line) </strong> </td>
      <td> <strong> IoU(Drivable area) </strong> </td>
      <td> <strong> FPS </strong> </td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> <strong> YOLOP </strong> </td>
      <td> 7.9 </td>
      <td> 31,763 </td>
      <td> 0.70 </td>
      <td> 0.262</td>
      <td> 0.91 </td>
      <td> 10 </td>
    </tr>
    <tr>
      <td> <strong> YOLOPv2 </strong></td>
      <td> 48.64 </td>
      <td> 38,955 </td>
      <td> 0.87 </td>
      <td> 0.27 </td>
      <td> 0.93 </td>
      <td> 41 </td>
    </tr>
    <tr>
      <td> <strong> Hybridnets </strong></td>
      <td> 13 </td>
      <td> 54,482 </td>
      <td> 0.85 </td>
      <td> 0.31</td>
      <td> 0.95 </td>
      <td> 12 </td>
    </tr>
  </tbody>
</table>

<strong> Graphical representation of comparison of models </strong>

![graphical_comparison](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/assets/117708050/50e9b6fe-977e-4376-9b4f-5908e4b82f7a)



## Visualization
model used in all three are trained on the BDD100k dataset.
![Comparison](https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/Lane_Detection_and_Drivable_area_segmentation_using_YOLOPv2/data/demo/together_video.gif).

A glimpse of the inference we obtained on our campus videos
<img src="https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/assets/125551038/0c3af87a-a80f-4b65-a9f2-c730fc8b6863" width = 1280 height = 180 >

## Model Quantization: 
In our pursuit of finding a balance between accuracy and computational efficiency, we explored the post-training quantization technique for one of the satisfactory models, YOLOP, which also boasts a simpler architecture compared to YOLOPv2. We chose YOLOP for quantization due to its smaller number of parameters and model size, making it more amenable to this process. In conclusion, the implementation of post-training quantization on YOLOP demonstrated its viability as an optimized solution for lane detection with limited computation power. This approach allows us to achieve near-comparable accuracy to the original model, YOLOP, while benefiting from reduced parameters and model size, thus making it well-suited for deployment in resource-constrained environments.

## Deployment and Future Improvements
After post-training static quantization we end up reducing our model size and get a balance between accuracy and computational efficiency. Now we are ready to deploy it on an edge computing device like te NVIDIA Jetson Xavier. In the future, we plan to deploy our lane detection pipeline on the NVIDIA Xavier platform, a powerful and energy-efficient system-on-a-chip (SoC) designed for edge computing and AI applications. The NVIDIA Xavier's advanced architecture and computational capabilities make it an ideal candidate for running deep learning models, even in real-time scenarios. The successful deployment on Xavier will pave the way for scalable and practical integration of our lane detection solution in various real-time applications.

<img src= "https://github.com/IVDC-Club-IIT-Indore/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/data/demo/jetson-xavier-nx-dev-kit-2c50-D%402x.jpg" width="300" height="200" />

## Refrerences : 
[1] MLND Capstone project for Udacity's Machine Learning Nanodegree, (2017), Github reposoitory, https://github.com/mvirgo/MLND-Capstone

[2] Pytorch Profiler ,PyTorch Recipes[ + ], https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

[3] Vision Transformers for Computer Vision[+],https://towardsdatascience.com/using-transformers-for-computer-vision-6f764c5a078b

[4] Chuan-en Lin (2018 , Dec. 17). “Tutorial: Build a lane detector”, Available: https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132.[Apr 06, 2019]\

[5] article : https://link.springer.com/article/10.1007/s11633-022-1339-y

[6]HybridNets model, Original Paper : https://arxiv.org/abs/2203.09035

[7] ibaiGorordo / ONNX-LSTR-Lane-Detection(2021), Github repository, https://github.com/ibaiGorordo/ONNX-LSTR-Lane-Detection
  
[8]CAIC-AD / YOLOPv2(2022), Github Repository, https://github.com/CAIC-AD/YOLOPv2
