import sys
import torch

from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions

sys.path.append('/Users/khoanguyen-cp/gmu/Marabou')

class ImprovedMarabouCoreDP():
  def solve(self, network_activation, model, postcondition):
    inputQuery = MarabouCore.InputQuery()
    layers_info = model.get_layers_info()

    for index, layer_info in enumerate(layers_info):
      layer = layer_info['layer']
      layer_name = layer_info['name']

      if layer_name == "model_input":
        inputQuery = self.__process_input_layer(layer_info, inputQuery)

      elif layer_name == "final_output" or index == len(layers_info) - 1:
        inputQuery = self.__process_output_layer(layer_info, inputQuery, postcondition)
      
      elif isinstance(layer, torch.nn.ReLU):
        layer_activation = network_activation[layer_name]
        inputQuery = self.__process_relu_hidden_layer(layer_info, layer_activation, inputQuery)

      elif isinstance(layer, torch.nn.Linear):
        inputQuery = self.__process_linear_hidden_layer(layer_info, inputQuery)

    ## Run Marabou to solve the query
    options = createOptions(verbosity=0)
    result = list(MarabouCore.solve(inputQuery, options, ""))
    result.append(inputQuery)
    return result
  

  def __process_input_layer(self, layer_info, inputQuery):
    input_features = list(range(0, layer_info["in_features"]))
    # print(f"{layer_info['name']} has variables {input_features}")
    inputQuery.setNumberOfVariables(len(input_features))
    inputQuery = self.__set_boundary_for_unconstrained_linear_vars(input_features, inputQuery)
    return inputQuery
  

  def __process_relu_hidden_layer(self, layer_info, layer_activation, inputQuery):
    num_neurons = layer_info["out_features"]
    num_of_vars = inputQuery.getNumberOfVariables()
    layer_vars = list(range(num_of_vars, num_of_vars + num_neurons))
    # print(f"{type(layer_info['layer']).__name__} layer {layer_info['name']} has variables {layer_vars}")

    inputQuery.setNumberOfVariables(num_of_vars + num_neurons)
    inputQuery = self.__set_boundary_for_relu_vars(layer_vars, layer_activation, inputQuery)
    inputQuery = self.__set_relu_constraints(layer_vars, inputQuery)
    return inputQuery
  

  def __process_linear_hidden_layer(self, layer_info, inputQuery):
    num_neurons = layer_info["out_features"]
    num_of_vars = inputQuery.getNumberOfVariables()
    layer_vars = list(range(num_of_vars, num_of_vars + num_neurons))
    # print(f"{type(layer_info['layer']).__name__} layer {layer_info['name']} has variables {layer_vars}")

    inputQuery.setNumberOfVariables(num_of_vars + num_neurons)
    inputQuery = self.__set_boundary_for_unconstrained_linear_vars(layer_vars, inputQuery)
    inputQuery = self.__set_linear_constraints(layer_vars, layer_info, inputQuery)
    return inputQuery
  

  def __process_output_layer(self, layer_info, inputQuery, postcondition):
    num_neurons = layer_info["out_features"]
    num_of_vars = inputQuery.getNumberOfVariables()
    layer_vars = list(range(num_of_vars, num_of_vars + num_neurons))
    # print(f"{type(layer_info['layer']).__name__} layer {layer_info['name']} has variables {layer_vars}")

    inputQuery.setNumberOfVariables(num_of_vars + num_neurons)
    inputQuery = self.__set_boundary_for_unconstrained_linear_vars(layer_vars, inputQuery)
    inputQuery = self.__set_linear_constraints(layer_vars, layer_info, inputQuery)
    inputQuery = self.__set_classification_constraints(layer_vars, postcondition, inputQuery)
    return inputQuery

  ########
  
  def __set_boundary_for_unconstrained_linear_vars(self, vars, inputQuery):
    for var in vars:
      inputQuery.setLowerBound(var, -100)
      inputQuery.setUpperBound(var, 100)
    return inputQuery
  
  
  def __set_boundary_for_relu_vars(self, vars, layer_activation, inputQuery):
    for idx, var in enumerate(vars):
      if layer_activation[idx] == "ON": # var > 0
        inputQuery.setLowerBound(var, 1e-6)
        inputQuery.setUpperBound(var, 100)

      elif layer_activation[idx] == "OFF":
        inputQuery.setLowerBound(var, 0)
        inputQuery.setUpperBound(var, 0)
        
      else: # neuron is unconstrained, can be on or off
        inputQuery.setLowerBound(var, 0)
        inputQuery.setUpperBound(var, 100)
        
    return inputQuery
  

  def __set_relu_constraints(self, layer_vars, inputQuery):
    # each relu step is accompanied by a preceding layer of the same size
    prev_layer_vars = [var - len(layer_vars) for var in layer_vars]
    # print(f":::RELU CONSTRAINTS::: layer_vars: {layer_vars} - prev_layers_vars: {prev_layer_vars}\n")
    for idx, relu_var in enumerate(layer_vars):
      corresponding_prev_var = prev_layer_vars[idx]
      MarabouCore.addReluConstraint(inputQuery, corresponding_prev_var, relu_var)
    return inputQuery
  

  def __set_linear_constraints(self, layer_vars, layer_info, inputQuery):
    # in dense layers, each neuron is connected to every neuron in the preceding layer
    # we can check the current layer's in_features to see the size of the preceding layer
    prev_layer_size = layer_info["in_features"]
    prev_layer_start_var = layer_vars[0] - prev_layer_size
    prev_layer_vars = list(range(prev_layer_start_var, prev_layer_start_var + prev_layer_size))
    # print(f":::LINEAR CONSTRAINTS::: layer_vars: {layer_vars} - prev_layers_vars: {prev_layer_vars}\n")

    # Ex: x2 = w0*x0 + w1*x1 + b
    # <=> w0*x0 + w1*x1 - x2 = -b
    layer = layer_info["layer"]
    for idx, var in enumerate(layer_vars):
      coefficients = layer.weight[idx]
      equation = MarabouCore.Equation()
      equation.addAddend(-1, var)
      for prev_idx, prev_var in enumerate(prev_layer_vars):
        equation.addAddend(coefficients[prev_idx], prev_var)
        
      scalar = 0.0 if layer.bias == None else (-layer.bias)
      equation.setScalar(scalar)
      inputQuery.addEquation(equation)

    return inputQuery
  

  def __set_classification_constraints(self, layer_vars, classification, inputQuery):
    # if we want to check for postcondition of a class, we have to encode its inverse.
    # Ex: given 3 classes, y1, y2, y3. If postcondition is y1, then we have to encode y != y1, i.e. y1 <= y2 or y1 <= y3.
    # I.e. We have to create disjunction pairs for classification_var and other vars in the output layer
    classification_var = layer_vars[classification]
    disjunction = []
    for var in layer_vars: 
      if var == classification_var: continue
      equation_type = MarabouCore.Equation.EquationType(2) # type 2 is Less than or equal (LE inequality)
      equation = MarabouCore.Equation(equation_type) 
      equation.addAddend(1, classification_var)
      equation.addAddend(-1, var)
      equation.setScalar(0)
      disjunction.append([equation])
      
    MarabouCore.addDisjunctionConstraint(inputQuery, disjunction)
    return inputQuery