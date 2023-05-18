import os
import torch
from torch import nn

class ProphecyPaperNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()    
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(2, 2, bias=False), # First hidden layer, with relu activation
      nn.ReLU(),
      nn.Linear(2, 2, bias=False), # Second hidden layer, with relu activation 
      nn.ReLU(),
    )
    self.final_output = nn.Linear(2, 2, bias=False)
    self.assign_weights()

  
  def forward(self, x):
    x = self.flatten(x)
    relu_stack_outputs = self.linear_relu_stack(x)
    logits = self.final_output(relu_stack_outputs)
    return logits
  
  
  def assign_weights(self):
    with torch.no_grad():
      self.linear_relu_stack[0].weight = nn.Parameter(torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=torch.float))
      self.linear_relu_stack[2].weight = nn.Parameter(torch.tensor([[0.5, -0.2], [-0.5, 0.1]], dtype=torch.float))
      self.final_output.weight = nn.Parameter(torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float))

  
  def attach_relu_activation_hook(self):
    # hook fn
    def relu_activation_hook(layer_name, result_storage):
      def hook(model, inputs, outputs):
        result_storage[layer_name] = ["ON" if val > 0 else "OFF" for val in outputs[0]]
      return hook
    
    handles = []
    activation_storage = {}
    for idx, layer in enumerate(self.linear_relu_stack):
      if isinstance(layer, torch.nn.ReLU):
        handle = layer.register_forward_hook(relu_activation_hook(idx, activation_storage))
        handles.append(handle)
    return handles, activation_storage
  
  
  def attach_layer_output_hook(self):
    # hook fn
    def layer_output_hook(layer_name, result_storage):
      def hook(model, inputs, outputs):
        result_storage[layer_name] = outputs.detach()
      return hook
    
    handles = []
    output_storage = {}
    for idx, layer in enumerate(self.linear_relu_stack):
      handle = layer.register_forward_hook(layer_output_hook(idx, output_storage))
      handles.append(handle)
    handle = self.final_output.register_forward_hook(layer_output_hook("final_output", output_storage))
    handles.append(handle)
    return handles, output_storage
  

  

  
