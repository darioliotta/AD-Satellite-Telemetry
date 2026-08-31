import torch
import torch.nn as nn

class CNN1DAnomalyClassifier(nn.Module):
    def __init__(self, in_channels=1, window_size=50):
        super(CNN1DAnomalyClassifier, self).__init__()
        self.name = 'CNN1D_Classifier'

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 50 -> 25

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 25 -> 12

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 12 -> 6
        
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 6, 64)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu_fc(self.fc1(x))
        logits = self.fc2(x)
        return logits