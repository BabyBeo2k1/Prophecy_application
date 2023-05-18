import sys
sys.path.append('/Users/khoanguyen-cp/gmu/Marabou')

from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions

class MarabouCoreDP():
  def solve(self, network_activations, model, postcondition):
    num_of_vars = 0
    inputQuery = MarabouCore.InputQuery()
    
    # INPUT CONSTRAINTS
    # Input layer isn't part of the network, so we'll use the first layer's in_features
    input_features = list(range(0, model.linear_relu_stack[0].in_features))
    num_of_vars += len(input_features)
    # print(f"input: {input_features}")
    inputQuery.setNumberOfVariables(num_of_vars)
    inputQuery = self._set_boundary_for_linear_vars(input_features, inputQuery)
    
    # HIDDEN LAYER CONSTRAINTS
    for relu_layer_idx, activation_values in network_activations.items():
      # each relu step is accompanied by a preceding linear layer of the same size
      relu_layer = model.linear_relu_stack[relu_layer_idx]
      linear_layer = model.linear_relu_stack[relu_layer_idx - 1]
      
      # constraints for linear layer
      linear_vars = list(range(num_of_vars, num_of_vars + linear_layer.out_features))
      num_of_vars += linear_layer.out_features
      # print(f"linear: {linear_vars}")
      inputQuery.setNumberOfVariables(num_of_vars)
      inputQuery = self._set_boundary_for_linear_vars(linear_vars, inputQuery)
      inputQuery = self._set_linear_constraints(linear_vars, linear_layer, inputQuery)
      
      # constraints for relu layer
      relu_vars = list(range(num_of_vars, num_of_vars + linear_layer.out_features))
      num_of_vars += linear_layer.out_features
      # print(f"relu: {relu_vars}")
      inputQuery.setNumberOfVariables(num_of_vars)
      inputQuery = self._set_boundary_for_relu_vars(relu_vars, activation_values, inputQuery)
      inputQuery = self._set_relu_constraints(relu_vars, inputQuery)
      
    # OUTPUT CONSTRAINTS 
    # if we want to check for postcondition of a class, we have to encode its inverse.
    # Ex: given 3 classes, y1, y2, y3. If postcondition is y1, then we have to encode y != y1, i.e. y1 <= y2 or y1 <= y3.
    layer = model.final_output
    layer_vars = list(range(num_of_vars, num_of_vars + layer.out_features))
    num_of_vars += layer.out_features
    # print(f"output: {layer_vars}")
    
    inputQuery.setNumberOfVariables(num_of_vars)
    inputQuery = self._set_boundary_for_linear_vars(layer_vars, inputQuery)
    inputQuery = self._set_linear_constraints(layer_vars, layer, inputQuery)
    inputQuery = self._set_classification_constraints(layer_vars, layer, postcondition, inputQuery)
    
    ## Run Marabou to solve the query
    # print("solving with Marabou...")
    options = createOptions(verbosity=0)
    return MarabouCore.solve(inputQuery, options, "")

  ##################################

  def _set_boundary_for_relu_vars(self, relu_vars, layer_activations, inputQuery):
    for idx, var in enumerate(relu_vars):
      if layer_activations[idx] == "ON": # var > 0
        inputQuery.setLowerBound(var, 1e-6)
        inputQuery.setUpperBound(var, 100)
        
      elif layer_activations[idx] == "OFF":
        inputQuery.setLowerBound(var, 0)
        inputQuery.setUpperBound(var, 0)
        
      else: # neuron is unconstrained, can be on or off
        inputQuery.setLowerBound(var, 0)
        inputQuery.setUpperBound(var, 100)
        
    return inputQuery
      
  
  def _set_boundary_for_linear_vars(self, linear_vars, inputQuery):
    for var in linear_vars:
      inputQuery.setLowerBound(var, -100)
      inputQuery.setUpperBound(var, 100)
    return inputQuery
  

  def _set_linear_constraints(self, layer_vars, layer, inputQuery):
    # in dense layers, each neuron is connected to every neuron in the preceding layer
    # we can check the current layer's in_features to see the size of the preceding layer
    prev_layer_vars = list(range(layer.out_features - layer.in_features, layer.in_features))
    
    # Ex: x2 = w0*x0 + w1*x1 + b
    # <=> w0*x0 + w1*x1 - x2 = -b
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


  def _set_relu_constraints(self, relu_vars, inputQuery):
    # each relu step is accompanied by a preceding linear layer of the same size
    linear_vars = [var - len(relu_vars) for var in relu_vars]
    for idx, relu_var in enumerate(relu_vars):
      corresponding_linear_var = linear_vars[idx]
      MarabouCore.addReluConstraint(inputQuery, corresponding_linear_var, relu_var)
    return inputQuery


  def _set_classification_constraints(self, layer_vars, layer, classification, inputQuery):
    # Create disjunction pairs for classification_var and other vars in the output layer
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