import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dropout_rate, input_size, hidden_sizes, output_size):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.dropout.append(nn.Dropout(dropout_rate))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.dropout.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer, dropout in zip(self.layers[:-1], self.dropout):
            x = layer(x)
            x = dropout(x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        return x