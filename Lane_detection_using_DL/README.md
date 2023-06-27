Lane Detection Approach for Autonomous Cars Using Deep Learning

## Data Preparation
the dataset are used from [dataset1](https://github.com/mvirgo/MLND-Capstone) and [dataset2](https://data.mendeley.com/datasets/t576ydh9v8/4). This files are converted into NumPy files and resized the second dataset in the same size (80×160×3) as the first dataset for mixing the two datasets. After that, the original and labeled images were imported as NumPy arrays and stored in two different Numpy files. And these two NumPy files were merged with the Numpy files of the previous dataset for generating a mixed dataset.

## Download
The mixed dataset utilized in this work can be found [here](https://drive.google.com/drive/folders/11R40PdKmEBYpntgvVmQFbsE7LfIK8UEv?usp=drive_link)

#Result
Lanes predicted using the model can be found [here](https://github.com/ampady06/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/model_training.ipynb)

# Key Files
1. [model_training.ipynb](https://github.com/ampady06/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/model_training.ipynb)-  this is the proposed lane detection model to train using that data.

2. [Trained_model.h5](https://github.com/ampady06/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/Trained_model.h5)-These are the final outputs from the above CNN. Note that if you train the file above the originals here will be overwritten! These get fed into the below.
3. [Lane_detection.py](https://github.com/ampady06/IITISoC-23-IVR1-LaneDetection-using-LimitedComputationPower/blob/main/Lane_detection_using_DL/Lane_detection.py)) -Using the trained model and an input video, this predicts the lane, and returns the original video with predicted lane lines drawn onto it.

# Citation
@article{sensors-22-05595,
  title={LLDNet: A Lightweight Lane Detection Approach for Autonomous Cars Using Deep Learning},
  author={Khan, M.A.-M.; Haque, M.F.; Hasan, K.R.; Alamjani, S.H.; Baz, M.; Masud, M.; Al-Nahid, A.},
  journal={Sensors},
  volume={22},
  year={2022},
}



