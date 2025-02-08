import numpy as np

#======================#
# 激勵函數
#======================#
def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    #有可能之後丟的不會是1D的向量資料
    # 如果 x 是 1D 向量
    if x.ndim == 1:
        max_x = np.max(x)
        shift_x = x - max_x
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x)
    else:
        # x 是 2D 或更高維，通常視為 (batch_size, num_classes)
        # 沿著最後一個軸做 softmax
        max_x = np.max(x, axis=-1, keepdims=True)
        shift_x = x - max_x
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

#======================#
# 損失函數
#======================#

def mse_loss(y_pred, y_true):
    return np.mean((y_true - y_pred)**2)

def binary_cross_entropy(y_pred, y_true, eps=1e-10):
    
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_pred, y_true, eps=1e-10):

    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    return -np.sum(y_true * np.log(y_pred))

class Network:
    def __init__(
        #預留激勵函數，先寫個sigmoid，後續可以擴充
        self,
        layer_sizes,
        init_weights_list=None,
        hidden_activation=sigmoid,
        output_activation=sigmoid
    ):
        self.layer_sizes = layer_sizes
        self.hidden_act = hidden_activation
        self.output_act = output_activation

        # 計算要多少的w跟b
        self.total_params = 0
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            self.total_params += in_dim * out_dim  # W
            self.total_params += out_dim           # b

        self.weights = []
        self.biases = []

        if init_weights_list is not None:
            # 確認資料o不ok
            if len(init_weights_list) != self.total_params:
                raise ValueError(
                    f"we need {self.total_params} weight，but you only have {len(init_weights_list)}!"
                )
            
            idx = 0
            for i in range(len(layer_sizes) - 1):
                in_dim = layer_sizes[i]
                out_dim = layer_sizes[i + 1]

                w_size = in_dim * out_dim
                w_vals = init_weights_list[idx : idx + w_size]
                idx += w_size
                W = np.array(w_vals).reshape((in_dim, out_dim))

                b_size = out_dim
                b_vals = init_weights_list[idx : idx + b_size]
                idx += b_size
                b = np.array(b_vals)

                self.weights.append(W)
                self.biases.append(b)
        else:
            # 隨便給個參數
            np.random.seed(6666) 
            for i in range(len(layer_sizes) - 1):
                in_dim = layer_sizes[i]
                out_dim = layer_sizes[i + 1]
                W = np.random.randn(in_dim, out_dim) * 0.5
                b = np.zeros((out_dim,))
                self.weights.append(W)
                self.biases.append(b)

    def forward(self, x):

        if x.ndim == 1:
            x = x.reshape(1, -1)        
        
        a = x
        #print("[Layer 0] input =", a)  # 輸入層 檢查用
        num_layers = len(self.layer_sizes) - 1
        
        for i in range(num_layers):
            W = self.weights[i]
            b = self.biases[i]
            z = a.dot(W) + b
            #print(f"[Layer {i+1}] W=\n{W}, b={b}") # 檢查用
            #print(f"[Layer {i+1}] z={z}") # 檢查用

            a = z
            #print(f"[Layer {i+1}] a={a}") # 檢查用

            # 先考慮之後的激活函數
            if i < num_layers - 1:
                a = self.hidden_act(z)
            else:
                a = self.output_act(z)        
        return a[0]  
