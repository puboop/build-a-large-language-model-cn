# **************************************
# --*-- coding: utf-8 --*--
# @Project    : Build-A-Large-Language-Model-CN
# @Time       : 2026/3/31 15:08
# @Author     : white
# @FileName   : 1-自注意力.py
# @Software   : PyCharm
# **************************************
# Listing 3.1 A compact self-attention class
import torch
import torch.nn as nn

d_in = 2
d_out = 6
inputs = torch.tensor(
    [[0.43, 0.15],  # Your     (x^1)
     [0.55, 0.87],  # journey  (x^2)
     [0.57, 0.85],  # starts   (x^3)
     [0.22, 0.58],  # with     (x^4)
     [0.77, 0.25],  # one      (x^5)
     [0.05, 0.80]]  # step     (x^6)
)


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


#  Listing 3.2 A self-attention class using PyTorch's Linear layers
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


# torch.manual_seed(123)
# sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(inputs))
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)  # A
example = torch.ones(6, 6)  # B
print(dropout(example))

# A 我们使用的dropout率为0.5
# B 创建一个由1组成的矩阵
