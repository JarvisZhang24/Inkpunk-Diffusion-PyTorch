import torch
from torch import nn
from torch.nn import functional as F

class Self_Attention(nn.Module):
     def __init__(self , heads , channels):
          super().__init__()
          self.heads = heads
          self.channels = channels
          self.hidden_channels = channels // heads
          self.to_keys = nn.Linear(channels , channels , bias=False)
          self.to_values = nn.Linear(channels , channels , bias=False)
          self.to_queries = nn.Linear(channels , channels , bias=False)
          self.to_out = nn.Linear(channels , channels)