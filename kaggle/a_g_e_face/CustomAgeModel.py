import torch
from torch.nn import Module
from torch import nn, sigmoid, round
import math
from torchsummary import summary


class CustomAgeModel(Module):
    def __init__(self, device):
        super(CustomAgeModel, self).__init__()
        self.device = device
        self.features_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=1), nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=1), nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 128, kernel_size=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Flatten(),
        ).to(device)

        self.age_regressor = nn.Sequential(
            nn.Linear(self.get_linear_input(), 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Dropout(p=0.2), nn.Linear(1024, 1)
        ).to(device)

        self.eth_classifier = nn.Sequential(
            nn.Linear(self.get_linear_input(), 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Dropout(p=0.2), nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(128, 5)
        ).to(device)

        self.gender_clssifier = nn.Sequential(
            nn.Linear(self.get_linear_input(), 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1)
        ).to(device)


    def forward(self, x):
        x = self.features_extractor(x)
        return {
            'ethnicity': self.eth_classifier(x),
            'gender': round(sigmoid(self.gender_clssifier(x))),
            'age': self.age_regressor(x)}


    def get_linear_input(self):
        temp_input = torch.rand(1,1,48,48).to(self.device)
        out_x = self.features_extractor(temp_input)
        size = out_x.size()

        return math.prod(size)


if __name__ == '__main__':
    myModel = CustomAgeModel('cpu')
    print(summary(myModel, (1,48,48)))



