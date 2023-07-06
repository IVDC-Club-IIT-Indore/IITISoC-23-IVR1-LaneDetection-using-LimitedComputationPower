

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization

# Define your custom model

batch_size = 150
epochs = 20
pool_size = (2, 2)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.batch_norm = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 60, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(60, 50, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(50, 40, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.2)
        self.conv4 = nn.Conv2d(40, 30, kernel_size=3, stride=1)
        self.dropout2 = nn.Dropout(0.2)
        self.conv5 = nn.Conv2d(30, 20, kernel_size=3, stride=1)
        self.dropout3 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(20, 10, kernel_size=3, stride=1)
        self.dropout4 = nn.Dropout(0.2)
        self.conv7 = nn.Conv2d(10, 5, kernel_size=3, stride=1)
        self.dropout5 = nn.Dropout(0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.ConvTranspose2d(5, 10, kernel_size=3, stride=1)
        self.dropout6 = nn.Dropout(0.2)
        self.deconv2 = nn.ConvTranspose2d(10, 20, kernel_size=3, stride=1)
        self.dropout7 = nn.Dropout(0.2)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.ConvTranspose2d(20, 30, kernel_size=3, stride=1)
        self.dropout8 = nn.Dropout(0.2)
        self.deconv4 = nn.ConvTranspose2d(30, 40, kernel_size=3, stride=1)
        self.dropout9 = nn.Dropout(0.2)
        self.deconv5 = nn.ConvTranspose2d(40, 50, kernel_size=3, stride=1)
        self.dropout10 = nn.Dropout(0.2)
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv6 = nn.ConvTranspose2d(50, 60, kernel_size=3, stride=1)
        self.final_layer = nn.ConvTranspose2d(60, 1, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.relu(self.conv3(x))
        x = self.dropout1(x)
        x = self.relu(self.conv4(x))
        x = self.dropout2(x)
        x = self.relu(self.conv5(x))
        x = self.dropout3(x)
        x = self.pool2(x)
        x = self.relu(self.conv6(x))
        x = self.dropout4(x)
        x = self.relu(self.conv7(x))
        x = self.dropout5(x)
        x = self.pool3(x)
        x = self.upsample1(x)
        x = self.relu(self.deconv1(x))
        x = self.dropout6(x)
        x = self.relu(self.deconv2(x))
        x = self.dropout7(x)
        x = self.upsample2(x)
        x = self.relu(self.deconv3(x))
        x = self.dropout8(x)
        x = self.relu(self.deconv4(x))
        x = self.dropout9(x)
        x = self.relu(self.deconv5(x))
        x = self.dropout10(x)
        x = self.upsample3(x)
        x = self.relu(self.deconv6(x))
        x = self.final_layer(x)
        return x

# Create an instance of your model
model = MyModel()

# Create random inputs for testing
inputs = torch.randn(1, 3, 80, 160)


# Profile the model
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

# Print the profiling results
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))