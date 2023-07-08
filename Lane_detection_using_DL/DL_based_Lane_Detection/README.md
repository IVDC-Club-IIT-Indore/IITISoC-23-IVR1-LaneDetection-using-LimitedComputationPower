## LANE DETECTION WITH DEEP LEARNING
I use a deep learning-based approach to improve upon lane detection. 
My final model uses a fully convolutional neural network to output an image of a predicted lane.

## DATASETS

## For fully convolutional network
The full training set of images used for this model can be downloaded from [here](https://www.dropbox.com/s/rrh8lrdclzlnxzv/full_CNN_train.p?dl=0) and 
the full set of 'labels' (which are just the 'G' channel from an RGB image of a re-drawn lane with an extra dimension added to make use in Keras easier) 
[here](https://www.dropbox.com/s/ak850zqqfy6ily0/full_CNN_labels.p?dl=0).

## Images with coefficient labels
The original training images with no flips or rotations (downsized to 80x160x3) can be found [here](https://www.dropbox.com/s/1bnp70bhaz5kma9/coeffs_train.p?dl=0) and the 
related coefficient labels (i.e. not the drawn lane labels, but the cofficients for a polynomial line) can be found [here](https://www.dropbox.com/s/ieulvrcooetrlmd/coeffs_labels.p?dl=0).

## NETWORK ARCHITECTURE
Neural network architecture consist of 7convolutional layers, 3 maxpolling layers, 3 upsampling layer and 6 deconvolution layers, built with Keras on top of Tensorflow. The neural network assumes the inputs to be road images in the shape of 80 x 160 x3 with the labels as 80 x 160 x 1.

## DOWNLOAD
The mixed dataset utilized in this work can be found [here](https://drive.google.com/drive/folders/10PHvTpcQVyxpXFE_-pcghTuJTo7uZpZb?usp=sharing).

## RESULT
Lanes predicted using this model along with the input videos can be found [here](https://drive.google.com/drive/folders/1CsjWxevyYMTNpWf55BZCu7_ucUSHx5cZ?usp=sharing).

## KEY FILES
1.[demo.py](Lane_detection_using_DL/DL_based_Lane_Detection/demo.py)- Pipeline that returns the output video with lanes predicted using the already trained model and input video.

2.[model_training](Lane_detection_using_DL/DL_based_Lane_Detection/model_training.py)- After downloading the training images and labels above, this is the fully convolutional neural network to train using that data.

3.[full_CNN_model.h5](Lane_detection_using_DL/DL_based_Lane_Detection/full_CNN_model.h5)- This is the final outputs from the above model training CNN.

4.[model_evaluation.py](Lane_detection_using_DL/DL_based_Lane_Detection/model_evaluation.py)- To evaluate the trained model using various metrics such as F1 score, recall, precision, IOU and dice coefficient. Basically it loads a model from an .h5 file, loads the data, makes predictions using the loaded model, and calculates the metrics for evaluation.

5.[cpu_&ram_usage.py](Lane_detection_using_DL/DL_based_Lane_Detection/cpu_&_ram_usage.py)- This profile the CPU activities during the inference of the PyTorch model and also measures the memory usage of the PyTorch model.

## PERFORMANCE METRICS
Precision, recall, IOU, F1 score and dice coefficient are used to evaluate the performance matrix.

References:

[1]MLND Capstone project for Udacity's Machine Learning Nanodegree, (2017), Github reposoitory, https://github.com/mvirgo/MLND-Capstone

[2] Pytorch Profiler ,PyTorch Recipes[ + ], https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
