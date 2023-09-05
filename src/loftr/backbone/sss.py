
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat


input = torch.randn(4284, 128)


print(repeat(input, 'n c -> n ww c', ww=25).shape)
 