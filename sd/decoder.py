import torch
from torch import nn
from torch.nn import functional as F


from attention import Self_Attention

class VAE_AttentionBlock(nn.Module):
     def __init__(self , in_channels):
          super().__init__()
          self.group_norm = nn.GroupNorm(32 , in_channels)
          self.attention = Self_Attention(1,in_channels)

     def forward(self , x:torch.Tensor) -> torch.Tensor:
          # x : (batch_size , features , height , width)

          residual = x

          n , c , h , w = x.shape

          # (batch_size , features , height , width) -> (batch_size , features , height * width)
          x = x.view(n , c , h * w)
          
          # (batch_size , features , height * width) -> (batch_size , height * width , features)
          x = x.transpose(-1,-2)

          # (batch_size , height * width , features) -> (batch_size , height * width , features)
          x = self.attention(x)

          # (batch_size , height * width , features) -> (batch_size , features , height * width)
          x = x.transpose(-1,-2)

          # (batch_size , features , height * width) -> (batch_size , features , height , width)
          x = x.view(n , c , h , w)

          #x = self.group_norm(x)

          x += residual

          return x



class VAE_ResidualBlock(nn.Module):
     def __init__(self , in_channels , out_channels):
          super().__init__()
          self.group_norm1 = nn.GroupNorm(32 , in_channels)


          self.conv1 = nn.Conv2d(in_channels , out_channels , kernel_size=3 , padding=1)
          
          self.group_norm2 = nn.GroupNorm(32 , out_channels)     

          self.conv2 = nn.Conv2d(out_channels , out_channels , kernel_size=3 , padding=1)

          if in_channels == out_channels:
               self.residual_layer = nn.Identity()
          else:
               self.residual_layer = nn.Conv2d(in_channels , out_channels , kernel_size=1 , padding=0)

     def forward(self , x:torch.Tensor) -> torch.Tensor:
          # x : (batch_size , in_channels , height , width)
          # residual : (batch_size , out_channels , height , width)
          residual = x
          x = self.group_norm1(x)
          x = F.silu(x)
          x = self.conv1(x)
          x = self.group_norm2(x)
          x = F.silu(x)
          x = self.conv2(x)
          residual = self.residual_layer(residual)
          return x + residual
          
          
               
          
          

          


          


