import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, initialization_method='xavier'):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.initialize_weights(initialization_method)

    def initialize_weights(self, method):
        if method == 'xavier':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        elif method == 'kaiming':
            nn.init.kaiming_uniform_(self.fc1.weight)
            nn.init.kaiming_uniform_(self.fc2.weight)
        elif method == 'random':
            nn.init.uniform_(self.fc1.weight, -0.5, 0.5)
            nn.init.uniform_(self.fc2.weight, -0.5, 0.5)
        elif method == 'normal':
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
