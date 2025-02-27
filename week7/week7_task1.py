import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

# data clenaning is finished in the previous.

print("------ task 1 ------")
class H_W_Data(Dataset):
    def __init__(self, csv_file):
        
        df = pd.read_csv(csv_file)        
        self.X = df.iloc[:, :-1].values  
        self.y = df.iloc[:, -1].values  
        # trans for Tensor
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]
    
class H_W_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=2):
        super(H_W_Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)
    
dataset = H_W_Data('h_w.csv')

# 以 8:2 拆分 train / validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 建立 DataLoader, batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

input_dim = dataset.X.shape[1]  # dataset.X = [樣本數, 特徵數]

model = H_W_Model(input_dim=input_dim, hidden_dim=2)
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 訓練模型
epochs = 30
for epoch in range(epochs):
    # --- Training ---
    model.train()  
    total_loss = 0
    for batch_x, batch_y in train_loader:
        # 預測
        pred = model(batch_x).squeeze()         
        # 計算損失
        loss = criterion(pred, batch_y)
        
        # 反向傳播與更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # --- Validation ---
    model.eval()  # 驗證模式（關閉 Dropout / BatchNorm 等）
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            pred = model(batch_x).squeeze()
            loss = criterion(pred, batch_y)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")



