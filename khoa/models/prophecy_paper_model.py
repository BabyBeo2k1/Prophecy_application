import torch
from torch import nn
from models.base_model import BaseModel

class ProphecyPaperNetwork(BaseModel):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(2, 2, bias=False), # First hidden layer, with relu activation
      nn.ReLU(),
      nn.Linear(2, 2, bias=False), # Second hidden layer, with relu activation 
      nn.ReLU(),
    )
    self.final_output = nn.Linear(2, 2, bias=False)
    self.assign_weights()

  
  def forward(self, x):
    relu_stack_outputs = self.linear_relu_stack(x)
    logits = self.final_output(relu_stack_outputs)
    return logits
  
  
  def assign_weights(self):
    with torch.no_grad():
      self.linear_relu_stack[0].weight = nn.Parameter(torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=torch.float))
      self.linear_relu_stack[2].weight = nn.Parameter(torch.tensor([[0.5, -0.2], [-0.5, 0.1]], dtype=torch.float))
      self.final_output.weight = nn.Parameter(torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float))

  

  

  
