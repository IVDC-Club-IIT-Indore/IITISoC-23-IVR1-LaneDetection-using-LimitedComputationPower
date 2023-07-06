# Lane_Detection
Lane detection using deep learning (Fully Connected CNN) and OpenCV

Earlier we took an approach of Lane detection using Traditional methods but that approach fails in case of curved roads and bad weather conditions. 
In this approach Deep Learning is used in Lane Detection using convolutional neural network.
The aim of this approach is for the end result to be more effective and faster than traditional computervision techniques.

## Dataset:
The dataset used for this approach is a BDD100K Lane Marking Dataset. The dataset consist of Images and label.

## Fully Connected Neural Net:
Neural network architecture consist of  7convolutional layers, 3 polling layers, 3 upsampling layer and 7 deconvolution layers, built with Keras
on top of Tensorflow. The neural net assumes the inputs to be road images in the shape of 80 x 160 x3 with the labels as 80 x 160 x 1. The output layer of deconvolution layer is a filled with the lane detected as a predicted image.

## Model Training:
 Before the model training starts, the dataset is divided into training set and avalidation set created using sklearn train_test_split method. The model is compiled with mean squared error as the loss function and trained over 20 epoch on the whole training dataset and validated on the validation dataset.

## Prediction:
In this phase the saved model is loaded along with its weights. The test image is preprocessed to be of the shape 80 x 160 x 3, as the model expect that as the input shape. Onces done the preprocessed image is given as an input to the models which predicts the output over which the filling of green color is added as an region of the lane detected.

## Result:
Lanes predicted using the model can be found [here](https://drive.google.com/drive/folders/1GLXM979Mzc7ynV-o_FGpnAXeukUacPMv?usp=drive_link)

## Key Files:
1. [model.py](https://github.com/WebWizard104/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/Feature/Lane_detection_using_DL/Lane_detection_CNN_model/model.py)- this is proposed lane detection model to train.
2. [model.h5](https://github.com/WebWizard104/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/Feature/Lane_detection_using_DL/Lane_detection_CNN_model/model.h5)- These are the final outputs from above CNN
3. [run.ipynb](https://github.com/WebWizard104/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/Feature/Lane_detection_using_DL/Lane_detection_CNN_model/run.ipynb)- this is used to test the model on images.
4. [test.py](https://github.com/WebWizard104/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/Feature/Lane_detection_using_DL/Lane_detection_CNN_model/test.py)- this is useed to test the model on input videos
5. [computation.py](https://github.com/WebWizard104/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/Feature/Lane_detection_using_DL/Lane_detection_CNN_model/computation.py)- this can be used to check computation effeciency of the model. On modifying model architecture and forward pass,it can be used for custom models also.

## Performance Metrics:
The performances metrics to be used for evaluating the build model are confusion matrix, Precision, recall, ROC curve, PR curve, Mean Squared Error(MSE) and mAP score.



**References :**\
[1] Chuan-en Lin (2018 , Dec. 17). “Tutorial: Build a lane detector”, Available: https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132.[Apr 06, 2019]\
[2] Aydin Ayanzadeh (2018 , Mar. 19). “Udacity Advance Lane-Detection of the Road in Autonomous Driving”, Available:https://medium.com/deepvision/udacity-advance-lane-detection-of-the-road-in-autonomous-driving-5faa44ded487.[Apr 06, 2019]\
[3]Udacity, self-driving-car, (2017), GitHub repository, https://github.com/udacity/self-driving-car.[Apr 06, 2019]
