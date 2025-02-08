from week4_tool import *
import numpy as np

# Task1
task1_weights_list = [0.5, 0.6, 0.2, -0.6, 0.3, 0.25, 0.8, 0.4, -0.5, 0.5, 0.6, -0.25]

task1_nn = Network(
    layer_sizes=[2, 2, 2],
    init_weights_list=task1_weights_list,
    hidden_activation=relu,
    output_activation=linear
)

x1 = np.array([1.5, 0.5])
e1 = np.array([0.8, 1])  

x2 = np.array([0, 1])
e2 = np.array([0.5, 0.5])

o1 = task1_nn.forward(x1)  
loss1 = mse_loss(o1, e1)

o2 = task1_nn.forward(x2) 
loss2 = mse_loss(o2, e2)

print("===== Task 1 answer  ====")
print("output =", o1, ", MSE =", loss1)
print("output =", o2, ", MSE =", loss2)

# Task2
task2_weights_list = [0.5, 0.6, 0.2, -0.6, 0.3, 0.25, 0.8, 0.4, -0.5]

task2_nn = Network(
    layer_sizes=[2, 2, 1],
    init_weights_list=task2_weights_list,
    hidden_activation=relu,
    output_activation=sigmoid
)

x1 = np.array([0.75, 1.25])
e1 = np.array([1])      

x2 = np.array([-1, 0.5])
e2 = np.array([0])

o1 = task2_nn.forward(x1)  
loss1 = binary_cross_entropy(o1, e1)

o2 = task2_nn.forward(x2)  
loss2 = binary_cross_entropy(o2, e2)

print("===== Task 2 answer  ====")
print("output =", o1, ", BCE =", loss1)
print("output =", o2, ", BCE =", loss2)

# Task3
task3_weights_list = [0.5, 0.6, 0.2, -0.6, 0.3, 0.25, 0.8, 0.5, 0.3, -0.4, 0.4, 0.75, 0.6, 0.5, -0.5]

task3_nn = Network(
    layer_sizes=[2, 2, 3],
    init_weights_list=task3_weights_list,
    hidden_activation=relu,
    output_activation=sigmoid
)

x1 = np.array([1.5, 0.5])
e1 = np.array([1, 0, 1])      

x2 = np.array([0, 1])
e2 = np.array([1, 1, 0])

o1 = task3_nn.forward(x1)  
loss1 = binary_cross_entropy(o1, e1)

o2 = task3_nn.forward(x2)  
loss2 = binary_cross_entropy(o2, e2)

print("===== Task 3 answer  ====")
print("output =", o1, ", BCE =", loss1)
print("output =", o2, ", BCE =", loss2)

# Task4
task4_weights_list = [0.5, 0.6, 0.2, -0.6, 0.3, 0.25, 0.8, 0.5, 0.3, -0.4, 0.4, 0.75, 0.6, 0.5, -0.5]

task4_nn = Network(
    layer_sizes=[2, 2, 3],
    init_weights_list=task4_weights_list,
    hidden_activation=relu,
    output_activation=softmax
)

x1 = np.array([1.5, 0.5])
e1 = np.array([1, 0, 0])      

x2 = np.array([0, 1])
e2 = np.array([0, 0, 1])

o1 = task4_nn.forward(x1)  
loss1 = categorical_cross_entropy(o1, e1)

o2 = task4_nn.forward(x2)  
loss2 = categorical_cross_entropy(o2, e2)

print("===== Task 4 answer  ====")
print("output =", o1, ", CCE =", loss1)
print("output =", o2, ", CCE =", loss2)
