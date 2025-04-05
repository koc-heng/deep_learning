import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Doc2Vec
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


#######################################################
# 載入 Doc2Vec 模型
#######################################################
doc2vec_model = Doc2Vec.load("my_doc2vec.model")
print("load to Doc2Vec model！")


#######################################################
# 讀 CSV 轉 (X_vectors, y_labels)
#######################################################
#測試用data測試用data
#data = pd.read_csv("test_data.csv")
data = pd.read_csv("tokenized_data.csv")  
data = data.dropna(subset=['tokenized_title'])  # 確認砍光空值

X_vectors = []
y_labels = []

for i, row in data.iterrows():
    tokens = row['tokenized_title'].split()
    vector = doc2vec_model.infer_vector(tokens)
    X_vectors.append(vector)
    y_labels.append(row['board'])  # class label

print(f"total: {len(X_vectors)}")


#######################################################
# 將文字標籤轉成整數 
#######################################################
label_set = list(set(y_labels))  
label_to_idx = {label: idx for idx, label in enumerate(label_set)}

y_indices = [label_to_idx[label] for label in y_labels]
num_classes = len(label_set)
print(f"label category: {num_classes}")


#######################################################
# split data
#######################################################
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors,
    y_indices,
    test_size=0.2,
    random_state=42
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")


#######################################################
# bulid Dataset and DataLoader
#######################################################
class VectorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor, y_tensor

train_dataset = VectorDataset(X_train, y_train)
test_dataset  = VectorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)


#######################################################
#  MLP 分類模型
#######################################################
class ClassifierNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # 輸出 logits + CrossEntropyLoss

input_dim = doc2vec_model.vector_size
hidden_dim = 64
output_dim = num_classes  

model = ClassifierNN(input_dim, hidden_dim, output_dim)

#######################################################
# 超參數 (Loss, Optimizer, epochs, device)
#######################################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

EPOCHS = 30  # 你可自行調整


#######################################################
# traning ，紀錄 Loss 與 Accuracy
#######################################################
train_losses = []
train_accs   = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 1. 清空梯度
        optimizer.zero_grad()
        # 2. forward
        outputs = model(batch_x)
        # 3. loss
        loss = criterion(outputs, batch_y)
        # 4. backward
        loss.backward()
        optimizer.step()

        # 計算訓練集的 loss
        total_loss += loss.item()

        # 計算訓練集的正確率
        _, pred = torch.max(outputs, dim=1)
        correct += (pred == batch_y).sum().item()
        total   += batch_y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    avg_acc  = correct / total

    train_losses.append(avg_loss)
    train_accs.append(avg_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")


#######################################################
# test and evaluate
#######################################################
model.eval()
test_correct = 0
test_total   = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        _, pred = torch.max(outputs, dim=1)
        test_correct += (pred == batch_y).sum().item()
        test_total   += batch_y.size(0)

test_acc = test_correct / test_total
print(f"\n[Final_test] Accuracy: {test_acc:.4f}")


#######################################################
# 畫 Loss 與 Accuracy
#######################################################
# (1) Loss vs Epoch
plt.plot(train_losses)
plt.title("Training Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# (2) Accuracy vs Epoch
plt.plot(train_accs)
plt.title("Training Accuracy vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()