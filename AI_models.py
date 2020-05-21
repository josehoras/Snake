
import torch
import torch.nn as nn

# Define our neural networks
class linear_net(nn.Module):
    def __init__(self):
        super(linear_net, self).__init__()
        self.linear1 = nn.Linear(400*400, 400)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(400, 4)

    def forward(self, input):
        # print(input.shape)
        # in_reshaped = torch.unsqueeze(input, 1)
        in_reshaped = input.reshape(-1, 400*400)
        hidden = self.linear1(in_reshaped)
        x_relu = self.relu(hidden)
        scores = self.linear2(x_relu)
        scores /= 10000
        return scores


class features_2_layers(nn.Module):
    def __init__(self, H=100):
        super(features_2_layers, self).__init__()
        self.linear1 = nn.Linear(12, H)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(H, 5)

    def forward(self, input):
        hidden1 = self.linear1(input)
        hidden1 = self.relu(hidden1)
        scores = self.linear2(hidden1)

        return scores


class features_linear_5_net(nn.Module):
    def __init__(self):
        super(features_linear_5_net, self).__init__()
        self.linear1 = nn.Linear(12, 100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 5)

    def forward(self, input):
        hidden1 = self.linear1(input)
        hidden1 = self.relu(hidden1)
        hidden2 = self.linear2(hidden1)
        hidden2 = self.relu(hidden2)
        scores = self.linear3(hidden2)
        return scores

class features_linear_big_net(nn.Module):
    def __init__(self, p=0.5):
        super(features_linear_big_net, self).__init__()
        self.linear1 = nn.Linear(12, 100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 5)
        self.dropout = nn.Dropout(p=p)

    def forward(self, input):
        hidden1 = self.linear1(input)
        hidden1 = self.relu(hidden1)
        hidden1 = self.dropout(hidden1)

        hidden2 = self.linear2(hidden1)
        hidden2 = self.relu(hidden2)
        hidden2 = self.dropout(hidden2)

        hidden3 = self.linear3(hidden2)
        hidden3 = self.relu(hidden3)
        hidden3 = self.dropout(hidden3)

        scores = self.linear4(hidden3)
        return scores

class conv_net(nn.Module):
    def __init__(self):
        super(conv_net, self).__init__()
        self.convolution = nn.Conv2d(1, 10, 5, stride=4)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(5)
        self.linear = nn.Linear(3610, 4)

    def forward(self, input):
        # print(input.shape)
        in_reshaped = torch.unsqueeze(input, 1)
        # in_reshaped = input.reshape(1, 1, screen_size[0], screen_size[1])
        x_conv = self.convolution(in_reshaped)
        x_relu = self.relu(x_conv)
        x_conv_out = self.maxpool(x_relu)
        # print("-: ", x_conv_out.shape)
        x_flat = x_conv_out.reshape(-1, 3610)
        scores = self.linear(x_flat)
        scores /= 10000
        return scores