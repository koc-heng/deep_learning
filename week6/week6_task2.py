from week6_tool import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('titanic_preprocessed.csv')
df["AgeUnder16"] = (df["Age"] < 16).astype(int)

#測試後把票價拿掉了
Y = df["Survived"].values 
X = df[["Pclass","Sex","AgeUnder16","Title","F_Size", "SibSp", "Parch"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=66
)
print("Train size=", len(X_train), " Test size=", len(X_test))

net = Network(
    layer_sizes=[7, 8, 1],
    activations=[relu, sigmoid]
)
loss_fn = BCELoss()

train_loss, test_acc = train_batch_model(
    net, loss_fn,
    X_train, y_train, 
    X_test, y_test,
    epochs=1000, 
    batch_size=100,
    lr=0.01
)

# 繪圖
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(train_loss)
plt.title("Train Loss"); plt.xlabel("epoch"); plt.ylabel("loss")

plt.subplot(1,2,2)
plt.plot([a*100 for a in test_acc])
plt.title("Test Accuracy(%)"); plt.xlabel("epoch"); plt.ylabel("acc %")
plt.tight_layout()
plt.show()

#看結果
final_test_acc = test_acc[-1]
print(f"Final Test Accuracy={final_test_acc*100:.2f}%")