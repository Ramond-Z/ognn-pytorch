import torch
import numpy as np
import os
import ocnn
import time
import matplotlib.pyplot as plt


from typing import *
from torch import nn
from torch.nn import functional as F
# from ocnn.nn import OctreeConv, OctreeDeconv
from ocnn.octree import Octree, Points
from octreed import OctreeD as DualOctree
from nn import GraphConv
from copy import deepcopy
from tqdm.auto import tqdm
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


def relative_error(a: torch.Tensor, b: torch.Tensor):
    return ((a - b).abs() / b.abs().clamp(min=1e-6)).mean().item(), ((a - b).abs() / b.abs().clamp(min=1e-6)).max().item()


def absolute_error(a: torch.Tensor, b: torch.Tensor):
    return (a - b).abs().mean().item(), (a - b).abs().max().item()


def load_points(n_instances=1):
    root = '/home/v-chuazhou/my_container/datasets/ABO/pc_500000'
    instances = os.listdir(root)[:n_instances]
    points = []
    for instance in instances:
        raw = np.load(os.path.join(root, instance))
        points.append(Points(torch.from_numpy(raw['points']), torch.from_numpy(raw['normals'])))
    return points


def build_octree(full_depth, depth, bs, points):
    def point2octree(point):
        octree = Octree(depth, full_depth)
        octree.build_octree(point)
        return octree
    n_instances = len(points)
    n_iter = int(np.ceil(n_instances / bs))
    results = []
    for idx in range(n_iter):
        end = min((idx + 1) * bs, n_instances)
        to_merge = points[bs * idx: end]
        to_merge = [point2octree(p) for p in to_merge]
        octree = ocnn.octree.merge_octrees(to_merge)
        octree.construct_all_neigh()
        results.append(octree)
    return results


def graph_allclose_check(g1, g2):
    assert g1.key_shift == g2.key_shift
    assert g1.depth_min == g2.depth_min
    assert g1.depth_max == g2.depth_max
    assert g1.nnum == g2.nnum         # the total node number

    assert (g1.batch_id is None and g2.batch_id is None) or torch.allclose(g1.batch_id, g2.batch_id)
    assert (g1.node_depth is None and g2.node_depth is None) or torch.allclose(g1.node_depth, g2.node_depth)
    assert (g1.child is None and g2.child is None) or torch.allclose(g1.child, g2.child)        # the octree node has children or not
    assert (g1.key is None and g2.key is None) or torch.allclose(g1.key, g2.key)          # the bits from the 58th is the node depth
    assert (g1.octree_mask is None and g2.octree_mask is None) or torch.allclose(g1.octree_mask, g2.octree_mask)  # used to pad zeros for non-leaf nodes

    assert (g1.edge_idx is None and g2.edge_idx is None) or torch.allclose(g1.edge_idx, g2.edge_idx)     # the edge index in (i, j)
    assert (g1.edge_dir is None and g2.edge_dir is None) or torch.allclose(g1.edge_dir, g2.edge_dir)


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

  def extra_repr(self) -> str:
    return ('in_channels={}, out_channels={}, n_edge_type={}, n_node_type={}, '
            'use_bias={}'.format(self.in_channels, self.out_channels,
             self.n_edge_type, self.n_node_type, self.use_bias))  # noqa


def benchmark_model(model, octrees, datas, depth, repeats, warmup=10):
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    model = model.cuda()
    for data, octree in zip(datas, octrees):
        # noise = torch.randn_like(data)
        # data, octree = data.cuda(), octree.cuda()
        for _ in range(warmup):
            h = model(data, octree, depth)
            loss = h.sum()
            # loss = (h * noise).sum()
            loss.backward()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for data, octree in zip(datas, octrees):
        data, octree = data.cuda(), octree.cuda()
        for _ in range(repeats):
            h = model(data, octree, depth)
            # loss = (h * noise).sum()
            loss = h.sum()
            loss.backward()
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 1024 ** 2
    elapsed = time.time() - start

    # end.record()
    # torch.cuda.synchronize()
    # time = start.elapsed_time(end)

    return (elapsed / repeats) * 1000., memory


def main():
    torch.set_printoptions(profile='full')
    in_channels = 12
    out_channels = 12
    bs = 8
    n_sample = bs * 1
    dtype = torch.float16
    octree_full_depth = 3
    octree_depth = 11
    kernel_size = [3,]
    stride = 1
    points = load_points(n_sample)
    octrees = build_octree(full_depth=octree_full_depth, depth=octree_depth, bs=bs, points=points)
    results = []
    depths = range(3, 10)
    for octree in octrees:
        octree = DualOctree(octree).cuda()
        for d in tqdm(depths):
            k = 1
            data = torch.randn((octree.graphs[d].nnum, 32)).cuda()
            conv1 = GraphConvMonteCarlo(32, 64, 7, 0, False, k, True)
            conv2 = GraphConv(32, 64, n_node_type=0)
            new_time, _ = benchmark_model(conv1, [octree], [data], d, 10)
            original_time, _ = benchmark_model(conv2, [octree], [data], d, 10)
            print('depth {}, new time: {:.4f}, original_time: {:.4f}'.format(d, new_time, original_time))
    #     result = dict()
    #     for depth in depths:
    #         graph = octree.graphs[depth]
    #         row, col  = graph.edge_idx
    #         index = row * 7 + graph.edge_type
    #         count = index.unique(return_counts=True)[1].bincount().view(-1)
    #         result[depth] = count
    #     results.append(result)
    # plt.figure(figsize=(8, 5))
    # for depth in depths:
    #     total = [result[depth] for result in results]
    #     total = torch.stack(total, dim=0).float().mean(dim=0).view(-1)
    #     total = total / total.sum()
    #     idx = total.nonzero().view(-1)
    #     data = total[idx]
    #     plt.plot(idx, data, marker='o', label='depth-{}'.format(depth))
    # plt.xlabel('reduce_count')
    # plt.ylabel('count')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    # plt.savefig('test.png')
if __name__ == '__main__':

    main()
    # N = 6518
    # cin = 34
    # cout = 64
    # data = torch.randn((N,cin), device='cuda', requires_grad=True)
    # neigh = torch.randint(-1, N, (N, 7), dtype=torch.long, device='cuda')
    # print((neigh < 0).sum())
    # weight = torch.randn((cout, 7, cin), device='cuda', requires_grad=True)
    # res = flex_gemm(data, weight, neigh)
    # loss = res.sum()
    # loss.backward()
    # print(weight.grad)
