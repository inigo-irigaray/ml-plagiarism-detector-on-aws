# torch imports
import torch.nn.functional as F
import torch.nn as nn


class BinaryClassifier(nn.Module):
    """
    Defines a neural network that performs binary classification.
    """
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_features, hidden_dim),
                                  nn.ReLU(),
                                  nn.Dropout(0.4),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Dropout(0.4),
                                  nn.Linear(hidden_dim, output_dim),
                                  nn.Sigmoid())
    
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        return self.model(x)
    