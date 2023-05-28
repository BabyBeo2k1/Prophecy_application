import pulp 
import torch


class UnderApproximationBox():
  def solve(self, input_property, attr_min, attr_max, model):
    layers_info = model.get_layers_info()

    # Prepare inputs and objective function
    num_of_inputs = len(attr_min)
    pulp_problem, pulp_inputs = self.__prepare_inputs_and_objective_fn(num_of_inputs, attr_min, attr_max)

    # Constraints for neurons in input_property
    for idx, layer_info in enumerate(layers_info):
      layer = layer_info['layer']
      layer_name = layer_info['name']

      if layer_name == "model_input": continue
      if layer_name not in input_property: continue

      if isinstance(layer, torch.nn.ReLU):
        prev_layer = layers_info[idx - 1]['layer']
        activation = input_property[layer_name]
        pulp_problem = self.__set_relu_constraints(layer_info, prev_layer, activation, pulp_problem, pulp_inputs)
        
    result = pulp_problem.solve()
    for v in pulp_problem.variables():
      print(v.name, "=", v.varValue)

    return pulp_problem, result


  def __set_relu_constraints(self, layer_info, prev_layer, activation, pulp_problem, pulp_inputs):
    for neuron in range(layer_info["in_features"]):
      coefficients = prev_layer.weight[neuron]
      neuron_activation = activation[neuron]

      expressions = []
      for idx, (d_lo, d_hi) in enumerate(pulp_inputs):
        if coefficients[idx] >= 0: # d_hi
          expressions.append(coefficients[idx] * d_hi)
        else: 
          expressions.append(coefficients[idx] * d_lo)

      bias = 0.0 if prev_layer.bias == None else (- prev_layer.bias[neuron])
      if neuron_activation == "ON":
        pulp_problem += (-(pulp.lpSum(expressions)) <= bias)
      elif neuron_activation == "OFF":
        pulp_problem += (pulp.lpSum(expressions) <= -bias)

    return pulp_problem

  
  def __prepare_inputs_and_objective_fn(self, num_of_inputs, attr_min, attr_max):
    inputs = []
    # set up input variables
    for i in range(0, num_of_inputs):
      d_hi = pulp.LpVariable(f"d_hi_{i}", lowBound=attr_min[i], upBound=attr_max[i])
      d_lo = pulp.LpVariable(f"d_lo_{i}", lowBound=attr_min[i], upBound=attr_max[i])
      inputs.append((d_lo, d_hi))

    # objective function
    problem = pulp.LpProblem("UnderApproximationBox", pulp.LpMaximize)
    problem += pulp.lpSum([d_hi - d_lo for d_lo, d_hi in inputs])
    return problem, inputs
