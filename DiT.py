import torch
import torch.nn as nn
from einops import rearrange
import math


"""
Patchify:
    - Input: (batch_size, channels, height, width)
    - Output: (batch_size, num_patches, patch_dim)
    - Patch dim is the number of channels * the number of pixels in the patch
    - Num patches is the number of patches in the image
    - Patch size is the size of the patch
    - Patchify is a function that takes an image and returns a tensor of patches
    - You can assume the image is square
    - Naive application of .reshape will not organize the pixels into the correct patches! You must do this manually (or with einops)
"""


class Patchify(nn.Module):
    def __init__(self, patch_size=8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: Implement Patchify
        self.patch_szie = patch_size

    def forward(self, x):
        # TODO: Implement Patchify
        # x: (batch_size, channels, height, width)
        # batch_size -> b, channels -> c, height -> (h p1), width -> (w p2)  
        p = self.patch_szie
        out = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)

        # out: (batch_size, num_patches, patch_dim)
        # batch_size -> b, num_patches -> (h w), p*p*channels -> (p1 p2 c)
        return out


"""
Unpatchify:
    - Input: (batch_size, num_patches, patch_dim)
    - Output: (batch_size, channels, height, width)
    - Patch dim is the number of channels * the number of pixels in the patch
    - Num patches is the number of patches in the image
    - Patch size is the size of the patch
    - Unpatchify is a function that takes a tensor of patches and returns an image
    - You can assume the image is square
"""


class Unpatchify(nn.Module):
    def __init__(self, patch_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: Implement Unpatchify
        self.patch_szie = patch_size

    def forward(self, x):
        # TODO: Implement Unpatchify
        p = self.patch_szie
        num_patches = x.shape[1]
        h = w = int(num_patches ** 0.5) # sq assumption 
        out = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p, p2=p, h=h, w=w)
        return out



'''
FeedForward:
    - Input: (batch_size, num_patches, hidden_dim)
    - Output: (batch_size, num_patches, hidden_dim)
    - Hidden dim is the dimension of the hidden state
    - Num patches is the number of patches in the image
    - FeedForward is a function that takes a tensor of patches and returns a tensor of patches
    - You can assume the image is square
    - Refer to Attention is all you need Section 3.3
'''
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, inner_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## TODO: Implement FeedForward
        ## Modules needed: 2x Linear, ReLU
        self.FFN = nn.Sequential (
            nn.Linear(hidden_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, hidden_dim)
        )


    def forward(self, x):
        ## TODO: Implement FeedForward
        return self.FFN(x)



'''
SelfAttention:
    - Input: (batch_size, num_patches, hidden_dim)
    - Output: (batch_size, num_patches, hidden_dim)
    - Hidden dim is the dimension of the hidden state
    - Num patches is the number of patches in the image
    - SelfAttention is a function that takes a tensor of patches and returns a tensor of patches
    - You can assume the image is square
    - Refer to Attention is all you need Section 3.2.1 
'''
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, inner_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## TODO: Implement SelfAttention
        ## Modules needed: 3x Linear
        self.dk = inner_dim

        # learnable parms 
        self.W_q = nn.Linear(hidden_dim, inner_dim)
        self.W_k = nn.Linear(hidden_dim, inner_dim)
        self.W_v = nn.Linear(hidden_dim, inner_dim)

    def forward(self, x):
        # getting K, Q, V by proj x through the linear layers
        Q = self.W_q(x) # e.g. x @ W_q.weight.T + W_q.bias
        K = self.W_k(x)
        V = self.W_v(x)

        #compute scores = Q @ K.T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)

        # apply softmax over last dim (normalizing each row indep)
        weights = torch.softmax(scores, dim=-1)

        # multiply by V
        return torch.matmul(weights, V)


'''
MultiHeadSelfAttn:
    - Input: (batch_size, num_patches, hidden_dim)
    - Output: (batch_size, num_patches, hidden_dim)
    - Hidden dim is the dimension of the hidden state
    - Num patches is the number of patches in the image
    - MultiHeadSelfAttn is a function that takes a tensor of patches and returns a tensor of patches
    - You can assume the image is square
    - Refer to Attention is all you need Section 3.2.2
'''

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## TODO: Implement MultiHeadSelfAttn, you must use the SelfAttention modules you implemented above
        ## Modules needed: num_heads x SelfAttention, Linear
        self.inner_dim = hidden_dim // num_heads

        # each head is its own SelfAttention
        self.heads = nn.ModuleList([
            SelfAttention(hidden_dim, self.inner_dim)
            for _ in range(num_heads)
        ])

        # learnable pram that operates on the concatenated output of all parallel attention heads

        # reason input & output dim are the same bc  the whole point of multi-head attention is to
        # enrich each token's representation, not change its size 
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        ## TODO: Implement MultiHeadSelfAttn
        # run each head on the same x
        head_outputs = [head(x) for head in self.heads] # (B, T, inner_dim) x num_heads

        # concat along last dim
        # e.g. (B, T, 64+64+64+64) = (B, T, 256) = (B, T, hidden_dim) 
        # bc inner_dim x num_heads = hidden_dim
        out = torch.cat(head_outputs, dim=-1)

        return self.out_proj(out) # (B, T, hidden_dim)



"""
DiTBlock:
    - Input: (batch_size, num_patches, hidden_dim)
    - Output: (batch_size, num_patches, hidden_dim)
    - Hidden dim is the dimension of the hidden state
    - Num patches is the number of patches in the image
    - DiTBlock is a block that takes a tensor of patches and returns a tensor of patches
    - You can assume the image is square
"""

class DiTBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads, ff_dim, time_emb_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## TODO: Implement DiTBlock
        ## Modules needed: 2x LN, MLP (can be implemented using nn.Sequential), FeedForward, MultiHeadSelfAttention
        ## You must zero initialize the linear layers in the MLP


    ## X is the input patches, cond is the time embedding
    def forward(self, x, cond):

        ## TODO: Implement DiTBlock, x is the input patches, cond is the time embedding

        ## Step 1: Get the parameters from the condition

        ## Step 2: Apply layer normalization

        ## Step 3: Apply scale and shift

        ## Step 4: Apply multi-head self-attention

        ## Step 5: Scale output

        ## Step 6: Add the original input

        ## Step 7: Apply layer normalization

        ## Step 8: Apply scale and shift

        ## Step 9: Apply feedforward

        ## Step 10: Scale output

        ## Step 11: Add residual connection


        return None


## (FOR FLOW MATCHING EC ONLY) Given for free!
class ContinuousTimestepEmbedder(nn.Module):
    def __init__(self, emb_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.view(t.size(0))
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiT(nn.Module):

    ## Given for free! Use this to get the positional embeddings for the patches
    def get_position_embedding(self, num_patches, patch_size, hidden_dim):
        grid_size = int(patch_size ** 0.5)
        dim = hidden_dim // 2
        
        pos = torch.arange(num_patches, dtype=torch.float32)
        r = pos // grid_size
        c = pos % grid_size
        
        omega = torch.arange(0, dim, 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / dim))
        
        out_r = r[:, None] @ omega[None, :]
        out_c = c[:, None] @ omega[None, :]
        
        row_emb = torch.zeros((1, num_patches, dim))
        col_emb = torch.zeros((1, num_patches, dim))
        
        row_emb[:, :, 0::2] = torch.sin(out_r)
        row_emb[:, :, 1::2] = torch.cos(out_r)
        col_emb[:, :, 0::2] = torch.sin(out_c)
        col_emb[:, :, 1::2] = torch.cos(out_c)
        
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)
        pos_emb = nn.Parameter(pos_emb, requires_grad=False)
        return pos_emb

    def __init__(self, patch_size, num_blocks, num_heads, ff_dim, time_emb_dim, num_timesteps, hidden_dim, num_patches, num_channels=3, training_type="ddpm", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## TODO: Implement DiT
        ## Modules needed: Patchify, Linear, Embedding, Positional Embedding (use provided get_position_embedding), num_blocks x DiTBlock, LayerNorm, Linear, Unpatchify
        ## You must zero initialize the final linear layer


    def forward(self, image, timestep):
        ## TODO: Implement DiT

        ## Step 1: Patchify the image

        ## Step 2: Project the patches to the hidden dimension

        ## Step 3: Add the positional encoding

        ## Step 4: Create the time embedding

        ## Step 5: Apply the DiT blocks

        ## Step 6: Apply the layer normalization

        ## Step 7: Project the hidden dimension back to the patch dimension

        ## Step 8: Unpatchify the image

        return None
