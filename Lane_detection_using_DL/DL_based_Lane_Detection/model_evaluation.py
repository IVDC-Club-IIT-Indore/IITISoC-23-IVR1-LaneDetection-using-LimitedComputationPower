import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def dsc(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def iou(y_true, y_pred):
    intersection = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    union = K.sum(K.round(K.clip(y_true + y_pred, 0, 1)))
    iou_score = intersection / (union + K.epsilon())
    return iou_score

# Load the trained model from the .h5 file
model = load_model(r"C:\iitisoc\models\full_CNN_model.h5")

# Load the data
train_images = np.load(r"C:\iitisoc\Datasets\image_mixed.npy").astype(np.float32)
labels = np.load(r"C:\iitisoc\Datasets\label_mixed.npy").astype(np.float32)
labels = labels / 255.0
train_images, labels = shuffle(train_images, labels)
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.3)

# Make predictions using the loaded model
y_pred = model.predict(X_val)
y_pred_binary = np.round(y_pred)

# Threshold the true labels
y_val_binary = np.round(y_val)

# Calculate F1 score
f1 = f1_m(y_val_binary.reshape(-1), y_pred_binary.reshape(-1))

# Calculate recall
recall = recall_m(y_val_binary.reshape(-1), y_pred_binary.reshape(-1))

# Calculate precision
precision = precision_m(y_val_binary.reshape(-1), y_pred_binary.reshape(-1))

# Calculate Dice coefficient
dice = dsc(y_val_binary.reshape(-1), y_pred_binary.reshape(-1))

# Calculate IoU
iou_score = iou(y_val_binary.reshape(-1), y_pred_binary.reshape(-1))

print("F1 score:", f1)
print("Recall:", recall)
print("Precision:", precision)
print("Dice coefficient:", dice)
print("IoU score:", iou_score)
