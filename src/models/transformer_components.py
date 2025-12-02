# src/models/transformer_components.py
import math
import torch
import copy
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

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model：模型输入维度（512）
        d_ff：隐藏层的维度（通常是4倍d_model=2048）
        """
        super(PositionwiseFeedForward,self).__init__()
        # 第一层：线性变换 + ReLU
        self.w_1 = nn.Linear(d_model, d_ff)
        # 第二层：线性变换
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() # 或者 GELU (BERT常用)

    def forward(self, x):
        # x: [Batch, Seq, d_model]
        
        # 1. 升维: 512 -> 2048
        # 公式: ReLU(xW1 + b1)W2 + b2
        inter = self.activation(self.w_1(x))
        inter = self.dropout(inter)
        
        # 2. 降维: 2048 -> 512
        output = self.w_2(inter)
        
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # 1. Self-Attention 模块
        self.self_attn = MultiHeadAttention(d_model, n_heads)

        # 2. Feed-Forward 模块
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 3. 两个Add & Norm模块
        # LayerNorm的参数是特征维度d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 4. Dropout (防止过拟合的神器)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [Batch, Seq, d_model]
        mask: [Batch, 1, 1, Seq] (用于遮挡 Pad)
        """
        # --- 子层1：Multi-Head Attention ---
        # 1. 计算Attention
        attn_output, _ = self.self_attn(x, x, x, mask)

        # 2. Add & Norm
        # 残差连接: x + Dropout(Sublayer(x))
        # 这里的顺序是 Post-Norm: Norm(x + Sublayer(x)) -> 原始论文写法
        # 现代很多模型(GPT)用 Pre-Norm: x + Sublayer(Norm(x))
        # 我们按经典的来：
        x = self.norm1(x + self.dropout(attn_output))

        # --- 子层 2: Feed-Forward ---
        # 1. 计算 FFN
        ff_output = self.feed_forward(x)
        
        # 2. Add & Norm
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
    
def get_clones(module, N):
    """
    一个辅助函数：克隆 N 个相同的层
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, N=6, dropout=0.1):
        """
        Args:
            vocab_size: 词表大小 (比如 30000)
            d_model: 向量维度 (512)
            n_heads: 多头数 (8)
            d_ff: FFN隐藏层维度 (2048)
            N: 堆叠层数 (通常是 6 或 12)
        """
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.N = N
        
        # 1. 词嵌入层 (Word Embedding)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. 位置编码 (Positional Encoding)
        self.pe = PositionalEncoding(d_model)
        
        # 3. 堆叠 N 层 EncoderLayer
        # 先实例化一个标准层
        base_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        # 然后克隆 N 份
        self.layers = get_clones(base_layer, N)
        
        # 4. 最终的规范化层 (可选，但推荐)
        self.norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        """
        src: [Batch, Seq_Len] (注意：这里输入的是整数索引，不是向量了)
        mask: [Batch, 1, 1, Seq_Len]
        """
        # 1. Embedding
        # [Batch, Seq] -> [Batch, Seq, Dim]
        x = self.embedding(src)
        
        # 2. Scale Embedding (这是一个 trick)
        # 论文中提到 embedding 需要乘以 sqrt(d_model)，让数值变大一点，
        # 以便和 Positional Encoding (在 -1 到 1 之间) 相加时，
        # 原始语义信息不会被位置信息淹没。
        x = x * math.sqrt(self.d_model)
        
        # 3. Add Position Encoding
        x = self.pe(x)
        x = self.dropout(x)
        
        # 4. Pass through N layers
        for layer in self.layers:
            x = layer(x, mask)
            
        # 5. Final Norm
        return self.norm(x)
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        Args:
            img_size：图片分辨率（224）
            patch_size：每个小方块的大小（16）
            in_chans：输入通道数（RGB图片是3）
            embed_dim：转换后的特征维度（768, 和BERT-Base一样）
        """
        super().__init__()
        self.img_size=img_size
        self.patch_size = patch_size

        # 计算 Patch 的总数量
        # (224 // 16) * (224 // 16) = 14 * 14 = 196
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # --- 核心魔法: 使用 Conv2d 做切片 ---
        # kernel_size=16, stride=16
        # 这意味着卷积核一次看 16x16 的区域，然后跳过 16 个像素看下一个
        # 输出通道 embed_dim 就是要把这个 Patch 压缩成的向量长度
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: [Batch, Channels, Height, Width] (例如：[32, 3, 224, 224])
        """
        # 1. 卷积投影
        # input: [B, 3, 224, 224]
        # output: [B, 768, 14, 14]
        x = self.proj(x)

        # 2. 展平 (Flatten)
        # 我们需要序列，不需要 grid
        # 从维度 2 (H) 开始展平
        # output: [B, 768, 196] (196 = 14*14)
        x = x.flatten(2)

        # 3. 维度交换（Transpose）
        # Transformer 需要的输入是 [Batch, Seq_Len, Dim]
        # 目前是 [B, Dim, Seq_Len]，所以要交换 Dim 和 Seq_Len
        # output: [B, 196, 768]
        x = x.transpose(1, 2)

        return x