import numpy as np

# ===== 激活函式 ===== #
def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ===== Loss ===== #
class MSELoss:
    #看全部loss
    def get_total_loss(self, outputs, expects):
        return np.mean((outputs - expects)**2)
    #題目用的
    def get_output_losses(self, outputs, expects):
        n = outputs.shape[0]
        return (2.0/n)*(outputs - expects)

class BCELoss:
    """
    BCE' = -(E/O) + ((1-E)/(1-O))
    """
    def __init__(self, eps=1e-12):
        self.eps = eps
    #看全部loss的部分
    def get_total_loss(self, outputs, expects):
        O_clip = np.clip(outputs, self.eps, 1-self.eps)
        return -np.sum(expects*np.log(O_clip) + (1-expects)*np.log(1-O_clip))
    #單一loss
    def get_output_losses(self, outputs, expects):
        O_clip = np.clip(outputs, self.eps, 1-self.eps)
        return - (expects / O_clip) + ((1-expects)/(1 - O_clip))
# ================== #

# ===== 神經網絡===== #
class Network:
    #給定layer 跟 激勵函數層
    def __init__(
        self,
        layer_sizes,
        activations,             
        init_weights_list=None
    ):
        self.layer_sizes = layer_sizes
        self.activations = activations

        #激勵函數給的要跟層數相等
        if len(activations) != (len(layer_sizes)-1):
            raise ValueError(f"activations have to  {len(layer_sizes)-1}, but you only have {len(activations)}!")

        # 計算總參數
        self.total_params = 0
        for i in range(len(layer_sizes)-1):
            in_dim  = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            self.total_params += (in_dim*out_dim + out_dim)

        self.weights = []
        self.biases  = []

        # ---- 初始化 ----
        if init_weights_list is not None:
            if len(init_weights_list) != self.total_params:
                raise ValueError(
                    f"Need {self.total_params} params, but got {len(init_weights_list)}!"
                )
            idx = 0
            for i in range(len(layer_sizes)-1):
                in_dim  = layer_sizes[i]
                out_dim = layer_sizes[i+1]
                w_size  = in_dim*out_dim

                w_vals  = init_weights_list[idx: idx+w_size]
                idx    += w_size
                W = np.array(w_vals).reshape((in_dim, out_dim))

                b_size = out_dim
                b_vals = init_weights_list[idx: idx+b_size]
                idx   += b_size
                b = np.array(b_vals)

                self.weights.append(W)
                self.biases.append(b)
        else:
            # 假定一個預設值怕沒給參數
            np.random.seed(666666666)
            for i in range(len(layer_sizes)-1):
                in_dim  = layer_sizes[i]
                out_dim = layer_sizes[i+1]
                W = np.random.randn(in_dim, out_dim)*0.1
                b = np.zeros(out_dim)
                self.weights.append(W)
                self.biases.append(b)

    def forward(self, x):
        #前向傳播, 每層用 self.activations[i](z)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self.z_vals = []
        self.a_vals = []

        a = x
        self.a_vals.append(a)  # 第 0 層(輸入)
        num_layers = len(self.layer_sizes)-1

        for i in range(num_layers):
            W = self.weights[i]
            b = self.biases[i]
            z = a.dot(W) + b

            self.z_vals.append(z)

            act_fn = self.activations[i]
            a = act_fn(z)
            
            self.a_vals.append(a)

        return a[0]
    
    def _apply_activation_deriv(self, dL_da, z_val, act_fn):

        if act_fn == linear:
            return dL_da
        elif act_fn == relu:
            mask = (z_val>0).astype(float)
            return dL_da * mask
        elif act_fn == sigmoid:
            sigz = sigmoid(z_val)
            return dL_da * sigz*(1 - sigz)
        else:
            return dL_da
        
    def zero_grad(self, lr=0.01):        
        #梯度下降更新        
        for i in range(len(self.weights)):
            self.weights[i] -= lr*self.dweights[i]
            self.biases[i]  -= lr*self.dbiases[i]

    def backward(self, dL_dO):        
        #依 self.activations[i] 做對應微分        
        self.dweights = [None]*len(self.weights)
        self.dbiases  = [None]*len(self.biases)

        # 最後一層
        z_out = self.z_vals[-1][0]  # shape=(out_dim,)
        act_fn_out = self.activations[-1]

        # 求輸出層對 z 的微分
        delta = self._apply_activation_deriv(dL_dO, z_out, act_fn_out)

        #dW, db
        A_prev = self.a_vals[-2] 
        dW = np.outer(A_prev.flatten(), delta)
        db = delta.copy()
        self.dweights[-1] = dW
        self.dbiases[-1]  = db

        # 向前
        delta_cur = delta
        for layer_idx in reversed(range(len(self.weights)-1)):
            W_next = self.weights[layer_idx+1]
            # dL/d a(l)
            dL_da_l = delta_cur @ W_next.T

            z_l = self.z_vals[layer_idx][0]
            act_fn_l = self.activations[layer_idx]
            delta_l = self._apply_activation_deriv(dL_da_l, z_l, act_fn_l)

            A_prev_l = self.a_vals[layer_idx]
            dW_l = np.outer(A_prev_l.flatten(), delta_l)
            db_l = delta_l.copy()

            self.dweights[layer_idx] = dW_l
            self.dbiases[layer_idx]  = db_l

            delta_cur = delta_l