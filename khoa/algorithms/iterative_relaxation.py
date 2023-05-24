

from algorithms.decision_procedure import MarabouCoreDP
import copy
import torch
from torch import nn

class IterativeRelaxation():
  def __init__(self):
    self.dp = MarabouCoreDP()

  def call(self, model, input_data, postcondition):
    # attach hooks to model to get activation signature of X
    _act_handles, activation_signature = model.attach_relu_activation_hook()
    _out_handles, layer_outputs = model.attach_layer_output_hook()

    # evaluate model with X to get activation signature of X
    X = torch.tensor(input_data, dtype=torch.float)
    _logits = model(X)

    status, _, _, _ = self.dp.solve(activation_signature, model, postcondition)
    if status == "sat":
      return [activation_signature, postcondition]
    
    layer_names = list(activation_signature.keys())
    max_unconstrained_layer_idx = len(layer_names) - 1
    unconstrained_layer_idx = max_unconstrained_layer_idx

    while unconstrained_layer_idx >= 0: 
      # print(f"unconstrained_layer_id: {unconstrained_layer_idx}")
      
      # unconstrain all neurons in the layer
      layer_name = layer_names[unconstrained_layer_idx]
      original_activation = activation_signature[layer_name]
      activation_signature[layer_name] = ["--" for val in original_activation]
      # print(activation_signature)
      
      status, _, _, _ = self.dp.solve(activation_signature, model, postcondition)
      
      if status == "sat": # critical layer found
        # print(f"Critical layer found: {unconstrained_layer_idx}")
        
        crit_layer_idx = unconstrained_layer_idx
        crit_layer_name = layer_names[crit_layer_idx]
        # add back activations from critical layer
        activation_signature[crit_layer_name] = copy.deepcopy(original_activation)
        crit_layer_activation = activation_signature[crit_layer_name]
        
        # iteratively unconstrain neurons to see if they are needed
        for neuron_idx, _val in enumerate(activation_signature[crit_layer_name]):
          crit_layer_activation[neuron_idx] = "--"
          # print(f"--- unconstraining neuron {neuron_idx} in critical layer")
          # print(activation_signature)
          status, _, _, _ = self.dp.solve(activation_signature, model, postcondition)
          
          if status == "sat": # neuron needed, must remain constrained
            # print(f"--- neuron needed")
            crit_layer_activation[neuron_idx] = original_activation[neuron_idx]
        
        return [activation_signature]
      
      else: 
        unconstrained_layer_idx -= 1
