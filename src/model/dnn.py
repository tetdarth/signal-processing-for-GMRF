import torch
import torch.nn as nn

class dnn(nn.Module):
    def __init__(self, n_input=100, n_output=4):
        super(dnn, self).__init__()
        self.n_elements = 400
        self.do_rate = 0.3
        self.n_input = n_input
        self.n_output = n_output

        self.dropout1 = nn.Dropout(self.do_rate)
        self.fc1 = nn.Linear(n_input, self.n_elements)
        self.bn1 = nn.BatchNorm1d(self.n_elements)
        self.relu1 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout(self.do_rate)
        self.fc2 = nn.Linear(self.n_elements, self.n_elements)
        self.bn2 = nn.BatchNorm1d(self.n_elements)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout3 = nn.Dropout(self.do_rate)
        self.fc3 = nn.Linear(self.n_elements, self.n_elements)
        self.bn3 = nn.BatchNorm1d(self.n_elements)
        self.relu3 = nn.ReLU(inplace=True)

        self.dropout4 = nn.Dropout(self.do_rate)
        self.fc4 = nn.Linear(self.n_elements, self.n_output)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.dropout4(x)
        x = self.fc4(x)

        return x