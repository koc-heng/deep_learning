import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class Titanic(Dataset):
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


dataset = Titanic('titanic_selected.csv')

# 以 8:2 拆分 train / validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 建立 DataLoader（批次大小可自行調整）
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

class TitanicModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(TitanicModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  
        )
        
    def forward(self, x):
        return self.net(x)
    
input_dim = dataset.X.shape[1] 
input_dim

# 初始化模型與損失函數
model = TitanicModel(input_dim=input_dim, hidden_dim=8)
criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 200
for epoch in range(epochs):
    # --- Training ---
    model.train()  
    total_loss = 0
    for batch_x, batch_y in train_loader:
       
        pred = model(batch_x).squeeze()        
 
        loss = criterion(pred, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    # --- Validation ---
    model.eval()  # 驗證模式
    total_val_loss = 0
    correct = 0  
    total = 0   

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x).squeeze()  # 輸出 logits
            loss = criterion(outputs, batch_y)
            total_val_loss += loss.item()
            
            #0.5 門檻得到預測的標籤
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct / total

    print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}  Val Acc: {val_accuracy:.4f}")