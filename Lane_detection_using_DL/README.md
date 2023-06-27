Lane Detection Approach for Autonomous Cars Using Deep Learning


## Data Preparation
The original images and the labeled images of the first dataset were converted into two different pickle files and uploaded to Mr. Michael Vigro’s GitHub page (https://github.com/mvirgo/MLND-Capstone). After downloading the pickle files, this work converted them into NumPy files. The second dataset was downloded from https://data.mendeley.com/datasets/t576ydh9v8/4. After downloading, we resized the second dataset in the same size (80×160×3) as the first dataset for mixing the two datasets. After that, the original and labeled images were imported as NumPy arrays and stored in two different Numpy files. And these two NumPy files were merged with the Numpy files of the previous dataset for generating a mixed dataset.

## Download
The mixed dataset utilized in this work can be found https://drive.google.com/file/d/1S23Ac0_hbOktV0rE2q0IkQWpQjUfkMTB/view?usp=sharing 
and https://drive.google.com/file/d/1I264WVBL3Dyp_4PTfEYkVIDkg_Yn5gJJ/view?usp=sharing

# Key Files
1. [LLDNet.ipynb](https://github.com/Masrur02/LLDNet/blob/main/LLDNet.ipynb)- Assuming you have downloaded the training images and labels above, this is the proposed LLDNet to train using that data.

2. [LLDNet.h5](https://github.com/Masrur02/LLDNet/blob/main/LLDNet.h5)-These are the final outputs from the above CNN. Note that if you train the file above the originals here will be overwritten! These get fed into the below.
3. [Lane_detection.py](https://github.com/Masrur02/LLDNet/blob/main/Lane_detection.py) -Using the trained model and an input video, this predicts the lane, and returns the original video with predicted lane lines drawn onto it.

# Citation
@article{sensors-22-05595,
  title={LLDNet: A Lightweight Lane Detection Approach for Autonomous Cars Using Deep Learning},
  author={Khan, M.A.-M.; Haque, M.F.; Hasan, K.R.; Alamjani, S.H.; Baz, M.; Masud, M.; Al-Nahid, A.},
  journal={Sensors},
  volume={22},
  year={2022},
}



