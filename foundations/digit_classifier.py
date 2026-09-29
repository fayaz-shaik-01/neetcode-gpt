import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        pass
        # Define the architecture here
        self.first_linear = nn.Linear(784,512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.second_linear = nn.Linear(512,10)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        output = self.sigmoid(self.second_linear(self.dropout(self.relu(self.first_linear(images)))))
        return torch.round(output,decimals=4)
        # Return the model's prediction to 4 decimal places
