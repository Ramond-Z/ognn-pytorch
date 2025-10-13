# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import math
import torch
import torch.nn.functional as F


from torch.utils.checkpoint import checkpoint
from ocnn.nn import OctreeGroupNorm
from octreed import OctreeD
from utils import scatter_mean


class Activation(torch.nn.Module):

  def __init__(self, act_type: str = 'relu', inplace: bool = True):
    super().__init__()
    self.act_type = act_type.lower()
    if self.act_type == 'relu':
      self.activation = torch.nn.ReLU(inplace)
    elif self.act_type == 'silu':
      self.activation = torch.nn.SiLU(inplace)
    elif self.act_type == 'gelu':
      self.activation = torch.nn.GELU()
    else:
      raise ValueError

  def forward(self, x: torch.Tensor):
    return self.activation(x)


class GraphConv(torch.nn.Module):

  def __init__(
          self, in_channels: int, out_channels: int, n_edge_type: int = 7,
          n_node_type: int = 0, use_bias: bool = False):
    super().__init__()
    self.avg_degree = 7
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_edge_type = n_edge_type
    self.n_node_type = n_node_type
    self.use_bias = use_bias

    node_channel = n_node_type if n_node_type > 1 else 0
    self.weights = torch.nn.Parameter(
        torch.Tensor(n_edge_type * (in_channels + node_channel), out_channels))
    if self.use_bias:
      self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    fan_in = self.avg_degree * self.in_channels
    fan_out = self.avg_degree * self.out_channels
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    torch.nn.init.uniform_(self.weights, -a, a)
    if self.use_bias:
      torch.nn.init.zeros_(self.bias)

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    graph = octree.graphs[depth]

    # concatenate the one_hot vector for node_type
    if self.n_node_type > 1:
      one_hot = F.one_hot(graph.node_type, num_classes=self.n_node_type)
      x = torch.cat([x, one_hot], dim=1)

    # x -> col_data
    row, col = graph.edge_idx
    index = row * self.n_edge_type + graph.edge_type
    col_data = scatter_mean(
        x[col], index, dim=0, dim_size=x.shape[0] * self.n_edge_type)

    # add self-loops
    index = torch.arange(graph.nnum, dtype=torch.int64, device=x.device)
    index = index * self.n_edge_type + (self.n_edge_type - 1)
    col_data[index] = x

    # matrix product
    output = col_data.view(x.shape[0], -1) @ self.weights

    # add bias
    if self.use_bias:
      out += self.bias
    return output

  def extra_repr(self) -> str:
    return ('in_channels={}, out_channels={}, n_edge_type={}, n_node_type={}, '
            'use_bias={}'.format(self.in_channels, self.out_channels,
             self.n_edge_type, self.n_node_type, self.use_bias))  # noqa

from flex_gemm.kernels.triton.spconv import sparse_submanifold_conv_fwd_implicit_gemm_splitk, sparse_submanifold_conv_bwd_implicit_gemm_splitk


def flex_gemm_forward_implicit(data: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, neighbour: torch.Tensor):
    return sparse_submanifold_conv_fwd_implicit_gemm_splitk(data, weight, bias, neighbour, -1)


def flex_gemm_backward_implicit(grad: torch.Tensor, weight: torch.Tensor, data: torch.Tensor, neighbour: torch.Tensor):
    return sparse_submanifold_conv_bwd_implicit_gemm_splitk(grad, data, weight, None, neighbour, -1)


class _flex_gemm(torch.autograd.Function):
    @staticmethod
    def forward(data, weight, neighbour):
        result = flex_gemm_forward_implicit(data, weight, None, neighbour)
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        data, weight, neighbour = inputs
        ctx.save_for_backward(data, weight, neighbour)

    @staticmethod
    def backward(ctx, upstream_grad):
        data, weight, neighbour = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            upstream_grad = upstream_grad.contiguous()
        g_data, g_weight, _ = flex_gemm_backward_implicit(upstream_grad, weight, data, neighbour)
        return g_data, g_weight, None


def flex_gemm(data, weight, neighbour):
    return _flex_gemm.apply(data, weight, neighbour)


def select_k_sorted(data: torch.Tensor, idx: torch.Tensor, k: int, max_idx: int = None):
    unique_idx, counts = torch.unique_consecutive(idx, return_counts=True)
    M = unique_idx.size(0)
    max_idx =  M + 1 if max_idx is None else max_idx
    D = data.size(-1)
    result = torch.zeros((max_idx, k, D), dtype=data.dtype, device=data.device)
    starts = torch.cat([torch.zeros((1,), device=idx.device, dtype=torch.long), counts.cumsum(0)[:-1]])
    rel_idx = torch.randint(0, counts.max(), (M, k), device=idx.device) % counts.unsqueeze(1)
    global_idx = starts.unsqueeze(1) + rel_idx
    result[unique_idx, :, :] = data[global_idx]
    return result


def select_k_indices_sorted(idx: torch.Tensor, k: int, max_idx: int):
   unique_idx, counts = torch.unique_consecutive(idx, return_counts=True)
   results = torch.ones((max_idx, k), device=idx.device, dtype=torch.long) * -1
   starts = torch.cat([torch.zeros((1, ), device=idx.device, dtype=torch.long), counts.cumsum(0)[:-1]])
   rel_idx = torch.randint(0, counts.max(), (unique_idx.size(0), k), dtype=torch.long, device=idx.device) % counts.unsqueeze(1)
   global_idx = starts.unsqueeze(1) + rel_idx
   results[unique_idx, :] = global_idx // 7
   return results


def select_k(data: torch.Tensor, idx: torch.Tensor, k: int, max_idx: int = None):
    sort_idx = torch.argsort(idx)
    return select_k_sorted(data[sort_idx], idx[sort_idx], k, max_idx)


def sort_by_key(data: torch.Tensor, key: torch.Tensor):
    sort_idx = torch.argsort(key)
    return data[sort_idx], key[sort_idx]


class GraphConvMonteCarlo(torch.nn.Module):

  def __init__(
          self, in_channels: int, out_channels: int, n_edge_type: int = 7,
          n_node_type: int = 0, use_bias: bool = False, n_samples: int = 2, use_triton: bool = False):
    super().__init__()
    self.avg_degree = 7
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_edge_type = n_edge_type
    self.n_node_type = n_node_type
    self.use_bias = use_bias
    self.n_samples = n_samples
    self.use_triton = use_triton

    node_channel = n_node_type if n_node_type > 1 else 0
    self.weights = torch.nn.Parameter(
        torch.randn(n_edge_type * (in_channels + node_channel), out_channels))
    if self.use_bias:
      self.bias = torch.nn.Parameter(torch.randn(out_channels))

  def forward(self, x: torch.Tensor, octree, depth: int):
    graph = octree.graphs[depth]

    # concatenate the one_hot vector for node_type
    if self.n_node_type > 1:
      one_hot = F.one_hot(graph.node_type, num_classes=self.n_node_type)
      x = torch.cat([x, one_hot], dim=1)

    # x -> col_data
    if not self.use_triton:
        row, col = graph.edge_idx
        index = row * self.n_edge_type + graph.edge_type
        col_data = select_k_sorted(
            x[col], index, self.n_samples, max_idx=x.shape[0] * self.n_edge_type).mean(1)

        # add self-loops
        index = torch.arange(graph.nnum, dtype=torch.int64, device=x.device)
        index = index * self.n_edge_type + (self.n_edge_type - 1)
        col_data[index] = x

        # matrix product
        output = col_data.view(x.shape[0], -1) @ self.weights
    else:
        assert self.n_samples == 1
        row, col = graph.edge_idx
        index = row * self.n_edge_type + graph.edge_type
        selected_neighbour = select_k_indices_sorted(index, self.n_samples, x.shape[0] * self.n_edge_type).view(-1)
        # print(selected_neighbour.shape)
        idx = torch.arange(graph.nnum, dtype=torch.long, device=x.device)
        idx = idx * self.n_edge_type + (self.n_edge_type - 1)
        selected_neighbour[idx] = torch.arange(graph.nnum, dtype=torch.long, device=x.device)
        selected_neighbour = selected_neighbour.reshape(-1, self.n_edge_type).contiguous()
        output = flex_gemm(x.contiguous(), self.weights.reshape(self.out_channels, self.n_edge_type, -1).contiguous(), selected_neighbour)

    # add bias
    if self.use_bias:
      output += self.bias
    return output


class GraphNorm(torch.nn.Module):

  def __init__(self, in_channels: int, norm_type: str = 'batch_norm'):
    super().__init__()

    self.in_channels = in_channels
    self.norm_type = norm_type

    self.group = 32
    self.min_group_channels = 4
    if self.min_group_channels * self.group > in_channels:
      self.group = in_channels // self.min_group_channels
    assert in_channels % self.group == 0

    if self.norm_type == 'batch_norm':
      self.norm = torch.nn.BatchNorm1d(in_channels)  # , bn_eps, bn_momentum)
    elif self.norm_type == 'group_norm':
      self.norm = OctreeGroupNorm(in_channels, self.group)
    else:
      raise ValueError

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    if self.norm_type == 'batch_norm':
      output = self.norm(x)
    elif self.norm_type == 'group_norm':
      output = self.norm(x, octree, depth)
    else:
      raise ValueError
    return output


class GraphConvNorm(torch.nn.Module):

  def __init__(
          self, in_channels: int, out_channels: int, n_edge_type: int = 7,
          n_node_type: int = 0, norm_type: str = 'batch_norm'):
    super().__init__()
    self.conv = GraphConv(in_channels, out_channels, n_edge_type, n_node_type)
    self.norm = GraphNorm(out_channels, norm_type)

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    out = self.conv(x, octree, depth)
    out = self.norm(out, octree, depth)
    return out


class GraphConvNormAct(torch.nn.Module):

  def __init__(
          self, in_channels: int, out_channels: int, n_edge_type: int = 7,
          n_node_type: int = 0, norm_type: str = 'batch_norm',
          act_type: str = 'relu'):
    super().__init__()
    self.conv = GraphConv(in_channels, out_channels, n_edge_type, n_node_type)
    self.norm = GraphNorm(out_channels, norm_type)
    self.act = Activation(act_type=act_type)

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    out = self.conv(x, octree, depth)
    out = self.norm(out, octree, depth)
    out = self.act(out)
    return out


class GraphPad(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    # pad zeros to get features for the original octree nodes
    octree_mask = octree.graphs[depth].octree_mask
    out = x.new_zeros(octree_mask.shape[0], x.shape[1])
    out[octree_mask] = x  # pad zeros for internal octree nodes
    return out


class Conv1x1(torch.nn.Module):

  def __init__(
          self, in_channels: int, out_channels: int, use_bias: bool = False):
    super().__init__()
    self.linear = torch.nn.Linear(in_channels, out_channels, use_bias)

  def forward(self, x):
    return self.linear(x)


class Conv1x1Norm(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int,
               norm_type: str = 'batch_norm'):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.norm = GraphNorm(out_channels, norm_type)
    # self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    out = self.conv(x)
    out = self.norm(out, octree, depth)
    # out = self.bn(out) #, octree, depth)
    return out


class Conv1x1NormAct(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int,
               norm_type: str = 'batch_norm', act_type: str = 'relu'):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.norm = GraphNorm(out_channels, norm_type)
    self.act = Activation(act_type)

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    out = self.conv(x)
    out = self.norm(out, octree, depth)
    out = self.act(out)
    return out


class Prediction(torch.nn.Module):

  def __init__(self, in_channels: int, mid_channels: int, out_channels: int,
               norm_type: str = 'batch_norm', act_type: str = 'relu'):
    super().__init__()
    self.conv1 = Conv1x1NormAct(in_channels, mid_channels, norm_type, act_type)
    self.conv2 = Conv1x1(mid_channels, out_channels, use_bias=True)

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    x = self.conv1(x, octree, depth)
    x = self.conv2(x)
    return x


class Upsample(torch.nn.Module):

  def __init__(self, in_channels: int):
    super().__init__()
    self.in_channels = in_channels
    self.weights = torch.nn.Parameter(torch.Tensor(in_channels, in_channels, 8))
    torch.nn.init.xavier_uniform_(self.weights)
    # TODO: add bias

  def forward(self, x: torch.Tensor):
    out = x @ self.weights.flatten(1)
    out = out.view(-1, self.in_channels)
    return out

  def extra_repr(self):
    return 'in_channels={}'.format(self.in_channels)


class Downsample(torch.nn.Module):

  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels

    self.weights = torch.nn.Parameter(torch.Tensor(in_channels, in_channels, 8))
    torch.nn.init.xavier_uniform_(self.weights)

  def forward(self, x: torch.Tensor):
    weights = self.weights.flatten(1).t()
    out = x.view(-1, self.in_channels * 8) @ weights
    return out

  def extra_repr(self):
    return 'in_channels={}'.format(self.in_channels)


class GraphDownsample(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int,
               norm_type: str = 'batch_norm', act_type: str = 'relu'):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.downsample = Downsample(in_channels)
    if in_channels != out_channels:
      self.conv1x1 = Conv1x1NormAct(
          in_channels, out_channels, norm_type, act_type)

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    # downsample nodes at layer depth
    numd = octree.nnum[depth]
    outd = self.downsample(x[-numd:])

    # get the nodes at layer (depth-1)
    out = x.new_zeros(octree.nnum[depth-1], x.shape[1])
    leaf_mask = octree.children[depth-1] < 0
    leaf_num = octree.lnum[depth-1]
    out[leaf_mask] = x[-leaf_num-numd:-numd]
    out[leaf_mask.logical_not()] = outd

    # construct the final output
    out = torch.cat([x[:-leaf_num-numd], out], dim=0)

    if self.in_channels != self.out_channels:
      out = self.conv1x1(out, octree, depth - 1)
    return out

  def extra_repr(self):
    return 'in_channels={}, out_channels={}'.format(
        self.in_channels, self.out_channels)


class GraphUpsample(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int,
               norm_type: str = 'batch_norm', act_type: str = 'relu'):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.upsample = Upsample(in_channels)
    if in_channels != out_channels:
      self.conv1x1 = Conv1x1NormAct(
          in_channels, out_channels, norm_type, act_type)

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    # upsample nodes at layer depth
    numd = octree.nnum[depth]
    leaf_mask = octree.children[depth] < 0
    outd = x[-numd:]
    out1 = self.upsample(outd[leaf_mask.logical_not()])

    # construct the final output
    out = torch.cat([x[:-numd], outd[leaf_mask], out1], dim=0)
    if self.in_channels != self.out_channels:
      out = self.conv1x1(out, octree, depth + 1)
    return out

  def extra_repr(self):
    return 'in_channels={}, out_channels={}'.format(
        self.in_channels, self.out_channels)


class GraphResBlock2(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int, n_edge_type: int = 7,
               n_node_type: int = 0, norm_type: str = 'batch_norm',
               act_type: str = 'relu', bottleneck: int = 4):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.bottleneck = bottleneck
    channel_m = int(out_channels / bottleneck)

    self.conva = GraphConvNormAct(
        in_channels, channel_m, n_edge_type, n_node_type, norm_type, act_type)
    self.convb = GraphConvNorm(
        channel_m, out_channels, n_edge_type, n_node_type, norm_type)
    if self.in_channels != self.out_channels:
      self.conv1x1 = Conv1x1Norm(in_channels, out_channels, norm_type)
    self.act = Activation(act_type)

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    x1 = self.conva(x, octree, depth)
    x2 = self.convb(x1, octree, depth)

    if self.in_channels != self.out_channels:
      x = self.conv1x1(x, octree, depth)

    out = self.act(x2 + x)
    return out


class GraphResBlock(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int, n_edge_type: int = 7,
               n_node_type: int = 0, norm_type: str = 'batch_norm',
               act_type: str = 'relu', bottleneck: int = 4):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.bottleneck = bottleneck
    channel_m = int(out_channels / bottleneck)

    self.conv1x1a = Conv1x1NormAct(in_channels, channel_m, norm_type, act_type)
    self.conv = GraphConvNormAct(
        channel_m, channel_m, n_edge_type, n_node_type, norm_type, act_type)
    self.conv1x1b = Conv1x1Norm(channel_m, out_channels, norm_type)
    if self.in_channels != self.out_channels:
      self.conv1x1c = Conv1x1Norm(in_channels, out_channels, norm_type)
    self.act = Activation(act_type)

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    x1 = self.conv1x1a(x, octree, depth)
    x2 = self.conv(x1, octree, depth)
    x3 = self.conv1x1b(x2, octree, depth)

    if self.in_channels != self.out_channels:
      x = self.conv1x1c(x, octree, depth)

    out = self.act(x3 + x)
    return out


class GraphResBlocks(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int, n_edge_type: int = 7,
               n_node_type: int = 0, norm_type: str = 'batch_norm',
               act_type: str = 'relu', bottleneck: int = 4, resblk_num: int = 1,
               resblk_type: str = 'basic', use_checkpoint: bool = True):
    super().__init__()
    self.resblk_num = resblk_num
    self.use_checkpoint = use_checkpoint
    channels = [in_channels] + [out_channels] * resblk_num
    ResBlk = self._get_resblock(resblk_type)
    self.resblks = torch.nn.ModuleList([ResBlk(channels[i], channels[i+1],
        n_edge_type, n_node_type, norm_type, act_type, bottleneck)
        for i in range(self.resblk_num)])  # noqa

  def _get_resblock(self, resblk_type):
    if resblk_type == 'bottleneck':
      return GraphResBlock
    elif resblk_type == 'basic':
      return GraphResBlock2
    else:
      raise ValueError

  def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
    for i in range(self.resblk_num):
      if self.use_checkpoint and self.training:
        x = checkpoint(self.resblks[i], x, octree, depth, use_reentrant=False)
      else:
        x = self.resblks[i](x, octree, depth)
    return x
