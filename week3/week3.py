from week3_tool import *
import numpy as np

# Task1
layer_sizes = [2,2,1]
my_weights = [0.5, 0.6, 0.2, -0.6, 0.3, 0.25, 0.8, 0.4, -0.5]
nn = Network(layer_sizes, init_weights_list=my_weights)

print("===== Task 1 answer  ====")
output_1 = nn.forward(np.array([1.5, 0.5]))
print(f"A: {output_1}")
outputs_2 = nn.forward(np.array([0, 1]))
print(f"A: {outputs_2}")  

# Task2
layer_sizes = [2,2,1,2]
my_weights = [0.5, 0.6, 1.5, -0.8, 0.3, 1.25, 0.6, -0.8, 0.3, 0.5, -0.4, 0.2, 0.5 ]
nn = Network(layer_sizes, init_weights_list=my_weights)

print("===== Task 2 answer  ====")
output_3 = nn.forward(np.array([0.75, 1.25]))
print(f"A: {output_3}")
output_4 = nn.forward(np.array([-1, 0.5]))
print(f"A: {output_4}")  

