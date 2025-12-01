# src/models/transformer_components.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Day 1: 位置编码 (Positional Encoding)
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 1. 初始化矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 2. 计算 div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 3. 填充 sin/cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 4. 增加 batch 维度并注册
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        # 这里的切片是为了适配输入序列的实际长度
        x = x + self.pe[:, :x.size(1), :]
        return x

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    计算Scaled Dot-Product Attention
    
    Args:
        query:  [batch_size, n_heads, len_q, d_k](为了通用，我们在这里假设又多头参数，单头也是一样的逻辑)
        key:    [batch_size, n_heads, len_k, d_k]
        value:  [batch_size, n_heads, len_v, d_v]

    Returns:
        output: [batch_size, n_heads, len_q, d_v]
        attn_weights: [batch_size, n_heads, len_q, len_k]
    """

    # 1. 获取d_k（用于缩放）
    d_k = query.size(-1)

    # 2. 计算 QK^T (Matmul)
    # query: [..., len_q, d_k]
    # key.transpose(-2, -1): [..., d_k, len_k] 将最后两个维度交换，做转置
    # scores: [..., len_q, len_k]
    scores = torch.matmul(query, key.transpose(-2, -1))

    #3. 缩放（Scale）
    # 这里的数学意义是防止点积结果过大导致 Softmax 梯度消失
    scores = scores / math.sqrt(d_k)
    
    # 4. Mask (可选)
    # 如果有 mask，把 mask 为 0 的位置的分数设为负无穷大 (-1e9)
    # 这样 Softmax 之后，这些位置的概率就会变成 0
    if mask is not None:
        # mask == 0的地方，填上-1e9
        scores = scores.masked_fill(mask == 0, -1e9)

    # 5. Softmax 归一化
    # dim=-1 表示对每一行的最后一个维度(即针对所有 key)做归一化
    attn_weights = F.softmax(scores, dim=-1)

    # 6. 加权求和 (Weighted Sum)
    # attn_weights: [..., len_q, len_k]
    # value:        [..., len_k, d_v]
    # output:       [..., len_q, d_v]
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    Day 3: 多头注意力 (Multi-Head Attention)
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        """
        [Batch, Seq, D_model] -> [Batch, Heads, Seq, D_k]
        """
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear Projections + Split Heads
        Q = self.split_heads(self.w_q(q), batch_size)
        K = self.split_heads(self.w_k(k), batch_size)
        V = self.split_heads(self.w_v(v), batch_size)
        
        # 2. Scaled Dot-Product Attention (复用上面的函数)
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. Concat Heads
        # [Batch, Heads, Seq, D_k] -> [Batch, Seq, Heads, D_k]
        attention_output = attention_output.permute(0, 2, 1, 3)
        # Flatten: [Batch, Seq, D_model]
        attention_output = attention_output.contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Final Linear
        output = self.w_o(attention_output)
        
        return output, attention_weights