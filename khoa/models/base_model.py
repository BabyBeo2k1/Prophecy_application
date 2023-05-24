import torch
from torch import nn

class BaseModel(nn.Module):
  def attach_relu_activation_hook(self):
    # hook fn
    def relu_activation_hook(layer_name, result_storage):
      def hook(_model, _inputs, outputs):
        result_storage[layer_name] = ["ON" if val > 0 else "OFF" for val in outputs[0]]
      return hook
    
    handles = []
    activation_storage = {}
    for name, module in list(self.named_modules()):
      nested_modules = list(module.named_modules())[1:]
      if len(nested_modules) > 0: continue # only process the layers, ignore all other containers
      if isinstance(module, torch.nn.ReLU):
        handle = module.register_forward_hook(relu_activation_hook(name, activation_storage))
        handles.append(handle)  

    return handles, activation_storage
  

  def attach_layer_output_hook(self):
    # hook fn
    def layer_output_hook(layer_name, result_storage):
      def hook(_model, _inputs, outputs):
        result_storage[layer_name] = outputs.detach()
      return hook
    
    handles = []
    output_storage = {}
    for name, module in list(self.named_modules()):
      nested_modules = list(module.named_modules())[1:]
      if len(nested_modules) > 0: continue # only process the layers, ignore all other containers
      handle = module.register_forward_hook(layer_output_hook(name, output_storage))
      handles.append(handle)  
    return handles, output_storage
  
  
  def get_layers_info(model):
    result = []
    
    for name, module in list(model.named_modules()):
      nested_modules = list(module.named_modules())[1:]
      # only process the layers, ignore all other containers
      if len(nested_modules) > 0: continue

      layer_info = {}
      # add more blocks here to handle different types of layers
      if isinstance(module, torch.nn.ReLU):
        # We'll use the preceding layers to get the info needed for ReLU layer
        prev_layer_info = result[-1]
        layer_info = {
          "name": name,
          "in_features": prev_layer_info["out_features"], 
          "out_features": prev_layer_info["out_features"], 
          "layer": module,
        }
        
      elif isinstance(module, torch.nn.Linear):
        layer_info = { 
          "name": name,
          "in_features": module.in_features, 
          "out_features": module.out_features, 
          "layer": module,
        }

      # if the first layer to be processed, then add information about the model's input layer
      # using the first layer's info
      if len(result) == 0:
        input_layer_info = {
          "name": "model_input",
          "in_features": layer_info["in_features"], 
          "out_features": layer_info["in_features"], 
          "layer": None,
        }
        result.append(input_layer_info)
      
      result.append(layer_info)
    return result

  