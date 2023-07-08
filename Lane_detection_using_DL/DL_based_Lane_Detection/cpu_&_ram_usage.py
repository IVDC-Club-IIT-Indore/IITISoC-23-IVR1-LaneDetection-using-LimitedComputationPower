import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization
import torchvision.transforms as transforms
import cv2


# Define your custom model

batch_size = 150
epochs = 20
pool_size = (2, 2)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.batch_norm = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=0, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=0, stride=1)
        self.maxpool = nn.MaxPool2d(pool_size)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=0, stride=1)
        self.dropout1 = nn.Dropout2d(0.2)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=1)
        self.dropout2 = nn.Dropout2d(0.2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=0, stride=1)
        self.dropout3 = nn.Dropout2d(0.2)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=1)
        self.dropout4 = nn.Dropout2d(0.2)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1)
        self.dropout5 = nn.Dropout2d(0.2)
        self.upsample1 = nn.Upsample(scale_factor=pool_size)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=0, stride=1)
        self.dropout6 = nn.Dropout2d(0.2)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=0, stride=1)
        self.dropout7 = nn.Dropout2d(0.2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=0, stride=1)
        self.dropout8 = nn.Dropout2d(0.2)
        self.deconv4 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=0, stride=1)
        self.dropout9 = nn.Dropout2d(0.2)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=0, stride=1)
        self.dropout10 = nn.Dropout2d(0.2)
        self.upsample2 = nn.Upsample(scale_factor=pool_size)
        self.deconv6 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=0, stride=1)
        self.final_layer = nn.Conv2d(16, 1, kernel_size=3, padding=0, stride = 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.dropout1(x)
        x = self.relu(self.conv4(x))
        x = self.dropout2(x)
        x = self.relu(self.conv5(x))
        x = self.dropout3(x)
        x = self.relu(self.conv6(x))
        x = self.dropout4(x)
        x = self.relu(self.conv7(x))
        x = self.dropout5(x)
        x = self.upsample1(x)
        x = self.relu(self.deconv1(x))
        x = self.dropout6(x)
        x = self.relu(self.deconv2(x))
        x = self.dropout7(x)
        x = self.relu(self.deconv3(x))
        x = self.dropout8(x)
        x = self.relu(self.deconv4(x))
        x = self.dropout9(x)
        x = self.relu(self.deconv5(x))
        x = self.dropout10(x)
        x = self.upsample2(x)
        x = self.relu(self.deconv6(x))
        x = self.final_layer(x)

        return x
# Create an instance of your model
model = MyModel()


image = cv2.imread(r"C:\iitisoc\test_img3_input.png")
image = cv2.resize(image, (80,160))
transformer = transforms.ToTensor()
tensor_image = transformer(image)


# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)





with profile(activities=[ProfilerActivity.CPU],profile_memory = True, record_shapes=True) as prof:
    with record_function("model_inference"):
        model(tensor_image.unsqueeze(0).to(device))

print(prof.key_averages().table(sort_by="self_cpu_memory_usage",row_limit=10))









