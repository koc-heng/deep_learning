from week6_tool import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Task 1
df = pd.read_csv('gender-height-weight.csv')

#改類別
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

#換成向量
gender_data = df["Gender"].values  
height_data = df["Height"].values  
weight_data = df["Weight"].values

#標準化 身高體重
mean_h = height_data.mean()
std_h  = height_data.std()
height_std = (height_data - mean_h) / std_h

mean_w = weight_data.mean()
std_w  = weight_data.std()
weight_std = (weight_data - mean_w) / std_w

xs = np.column_stack([gender_data, height_std])
ys = weight_std

net1 = Network(
    layer_sizes=[2, 2, 1],
    activations=[relu, linear]
)
loss_fn = MSELoss()

loss_outcome = train_full_batch(net1, loss_fn, xs, ys, epochs=1000, learning_rate=0.01)

plt.plot(range(1,1001), loss_outcome)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

answer = std_w*loss_outcome[-1]**(1/2)
print(f"The answer is {answer} less than 15 pounds, so we finish the task.") 