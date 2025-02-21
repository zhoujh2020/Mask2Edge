from typing import Optional
import torch
from torch import nn, Tensor
from models.mask2former.position_encoding import PositionEmbeddingSine
# from .DWConv import DepthWiseConv
from models.utils.neck import DeformLayer


class Dwpart(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            DeformLayer(in_planes=256, out_planes=256)
        )
        self.multihead_attn = nn.MultiheadAttention(256, 8, dropout=0.0)
        self.pe_layer = PositionEmbeddingSine(128, normalize=True)
        self.query_embed = nn.Embedding(100, 256)
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x, query, attn_mask):
        bs, _, h, w = x.shape
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        pos = (self.pe_layer(x, None).flatten(2)).permute(2, 0, 1)  # hw b c
        out1 = self.block1(x)
        k = out1.flatten(2).permute(2, 0, 1)  # hw b c
        v = out1.flatten(2).permute(2, 0, 1)
        # crossattn: q: q b c, k: hw b c, v: hw b c
        output, output_weights = self.multihead_attn(query=self.with_pos_embed(query, query_embed),
                                                     key=self.with_pos_embed(k, pos),
                                                     value=v, attn_mask=attn_mask,
                                                     key_padding_mask=None)
        # output = query + output
        return output


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_tensors[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_tensors[0] * (1 - sigmoid_i)))


class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


if __name__ == '__main__':
    model = Dwpart()
    query = torch.randn(4, 100, 256)
    input = torch.randn(4, 256, 160, 160)
    output = model(input, query=query)
    print(output.shape)
