from week5_tool import *

#task1
task1_nn = Network(
        layer_sizes=[2,2,1,2],
        activations=[relu, linear, linear],
        init_weights_list=[0.5, 0.6, 0.2, -0.6, 0.3, 0.25, 0.8, -0.5, 0.6, 0.6, -0.3, 0.4, 0.75]
    )

mse_loss = MSELoss()
inputs = np.array([1.5, 0.5])
expects= np.array([0.8, 1.0])

# forward
outs = task1_nn.forward(inputs)
# loss
loss_val = mse_loss.get_total_loss(outs, expects)
dL_dO = mse_loss.get_output_losses(outs, expects)
# backward
task1_nn.backward(dL_dO)
# update
task1_nn.zero_grad(0.01)


print("=====[Task 1-1]=====")
print("Loss=",loss_val, " outs=", outs)
print("Weights:")
for i,(W,b) in enumerate(zip(task1_nn.weights, task1_nn.biases)):
    print(f" Layer {i}: W=\n{W}\n b={b}")


print("=====[Task 1-2]=====")
epochs = 1000
lr = 0.01
for epoch in range(epochs):
    outs = task1_nn.forward(inputs)
    loss = mse_loss.get_total_loss(outs, expects)
    dLO  = mse_loss.get_output_losses(outs, expects)
    task1_nn.backward(dLO)
    task1_nn.zero_grad(lr)

print(f"[Task 1-2] After {epochs} updates => final loss={loss:.10f}, outputs={outs}")

#task2
task2_nn = Network(
        layer_sizes=[2,2,1],
        activations=[relu, sigmoid],
        init_weights_list=[0.5, 0.6, 0.2, -0.6, 0.3, 0.25, 0.8, 0.4, -0.5]
    )

bce_loss = BCELoss()
inputs = np.array([0.75, 1.25])
expects= np.array([1.0])

# forward
outs = task2_nn.forward(inputs)
# loss
loss_val = bce_loss.get_total_loss(outs, expects)
dL_dO = bce_loss.get_output_losses(outs, expects)
# backward
task2_nn.backward(dL_dO)
# update
task2_nn.zero_grad(0.01)

print("=====[Task 2]=====")
print("Loss=",loss_val, " outs=", outs)
print("Weights:")
for i,(W,b) in enumerate(zip(task2_nn.weights, task2_nn.biases)):
    print(f" Layer {i}: W=\n{W}\n b={b}")


print("=====[Task 2]=====")
epochs = 1000
for epoch in range(epochs):
    outs = task2_nn.forward(inputs)
    loss = bce_loss.get_total_loss(outs, expects)
    dLO  = bce_loss.get_output_losses(outs, expects)
    task2_nn.backward(dLO)
    task2_nn.zero_grad(lr)

print(f"[Task 2-2] After {epochs} updates => final loss={loss:.10f}, outputs={outs}")