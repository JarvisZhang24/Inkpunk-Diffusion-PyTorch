import torch
from torch import nn
from torch.nn import functional as F

class Self_Attention(nn.Module):
     def __init__(self, heads, d_embedding, in_proj_bias=True, out_proj_bias=True):
          super().__init__()
          
          self.heads = heads
          self.d_embedding = d_embedding
          
          # 确保通道数能被头数整除
          assert d_embedding % heads == 0, f"channels {d_embedding} must be divisible by heads {heads}"
          
          self.head_dim = d_embedding // heads
          
          # 用于生成Q, K, V的线性层
          self.in_proj = nn.Linear(d_embedding, 3 * d_embedding, bias=in_proj_bias)
          
          # 输出投影层
          self.out_proj = nn.Linear(d_embedding, d_embedding, bias=out_proj_bias)
          
     def forward(self, x: torch.Tensor , causal_mask = False) -> torch.Tensor:
          # x: (batch_size, sequence_length, channels)
          batch_size, seq_len, d_embedding = x.shape
          
          # 生成Q, K, V
          # (batch_size, seq_len, d_embedding) -> (batch_size, seq_len, 3 * d_embedding)
          qkv = self.in_proj(x)
          
          # 分割成Q, K, V，每个都是 (batch_size, seq_len, d_embedding)
          q, k, v = qkv.chunk(3, dim=-1)
          
          # 重塑为多头格式: (batch_size, seq_len, heads, head_dim)
          q = q.view(batch_size, seq_len, self.heads, self.head_dim)
          k = k.view(batch_size, seq_len, self.heads, self.head_dim)
          v = v.view(batch_size, seq_len, self.heads, self.head_dim)
          
          # 转置为: (batch_size, heads, seq_len, head_dim)
          q = q.transpose(1, 2)
          k = k.transpose(1, 2)
          v = v.transpose(1, 2)
          
          # 计算注意力分数
          # (batch_size, heads, seq_len, head_dim) x (batch_size, heads, head_dim, seq_len)
          # -> (batch_size, heads, seq_len, seq_len)
          scale = self.head_dim ** -0.5
          attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
          
          if causal_mask:
               mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
               attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
          
          # 应用softmax
          attn_weights = F.softmax(attn_scores, dim=-1)
          
          # 应用注意力权重到V
          # (batch_size, heads, seq_len, seq_len) x (batch_size, heads, seq_len, head_dim)
          # -> (batch_size, heads, seq_len, head_dim)
          attn_output = torch.matmul(attn_weights, v)
          
          # 重塑回原始格式
          # (batch_size, heads, seq_len, head_dim) -> (batch_size, seq_len, heads, head_dim)
          attn_output = attn_output.transpose(1, 2)
          
          # 合并多头: (batch_size, seq_len, channels)
          attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_embedding)
          
          # 最终输出投影
          output = self.out_proj(attn_output)
          
          return output