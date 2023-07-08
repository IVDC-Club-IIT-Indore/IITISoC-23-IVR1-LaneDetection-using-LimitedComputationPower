## _*LANE DETECTION WITH DEEP LEARNING*_
I use a deep learning-based approach to improve upon lane detection. 
My final model uses a fully convolutional neural network to output an image of a predicted lane.

## _*DATASETS*_

## _For fully convolutional network_
The full training set of images used for this model can be downloaded from [here](https://www.dropbox.com/s/rrh8lrdclzlnxzv/full_CNN_train.p?dl=0) and 
the full set of 'labels' (which are just the 'G' channel from an RGB image of a re-drawn lane with an extra dimension added to make use in Keras easier) 
[here](https://www.dropbox.com/s/ak850zqqfy6ily0/full_CNN_labels.p?dl=0).

## _Images with coefficient labels_
The original training images with no flips or rotations (downsized to 80x160x3) can be found [here](https://www.dropbox.com/s/1bnp70bhaz5kma9/coeffs_train.p?dl=0) and the 
related coefficient labels (i.e. not the drawn lane labels, but the cofficients for a polynomial line) can be found [here](https://www.dropbox.com/s/ieulvrcooetrlmd/coeffs_labels.p?dl=0).

## _*DOWNLOAD*_
The mixed dataset utilized in this work can be found [here](https://drive.google.com/drive/folders/10PHvTpcQVyxpXFE_-pcghTuJTo7uZpZb?usp=sharing).

## _*RESULT*_
Lanes predicted using this model along with the input videos can be found [here](https://drive.google.com/drive/folders/1CsjWxevyYMTNpWf55BZCu7_ucUSHx5cZ?usp=sharing).

## _*KEY FILES*_
1.[model_training](Lane_detection_using_DL/DL_based_Lane_Detection/model_training.py)- After downloading the training images and labels above, this is the fully convolutional neural network to train using that data.
2.[demo.py](Lane_detection_using_DL/DL_based_Lane_Detection/demo.py)- Pipeline that returns the output video with lanes predicted using the already trained model and input video.
3.[full_CNN_model.h5](Lane_detection_using_DL/DL_based_Lane_Detection/full_CNN_model.h5)- This is the final outputs from the above model training CNN.
