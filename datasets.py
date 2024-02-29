import torch
from torch_geometric.utils import is_undirected, to_undirected
import os
from torch import Tensor, LongTensor
import dataset_loader
import utils

class BaseGraph:
    def __init__(self, x: Tensor, edge_index: LongTensor, edge_weight: Tensor,
                 y: Tensor):
        self.x = x
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.y = y
        self.num_classes = torch.unique(y).shape[0]
        self.num_nodes = x.shape[0]
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.to_undirected()

    def to_undirected(self):
        if not is_undirected(self.edge_index):
            self.edge_index, self.edge_weight = to_undirected(
                self.edge_index, self.edge_attr)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_attr.to(device)
        self.y = self.y.to(device)
        return self







def load_dataset(name: str):
    '''
    load dataset into a base graph format.
    '''
    savepath = f"./data/{name}.pt"
    if name in [ 'chameleon', 'film', 'squirrel']:
        if os.path.exists(savepath):
            bg = torch.load(savepath, map_location="cpu")
            return bg
        ds = dataset_loader.DataLoader(name)
        data = ds[0]
        data.num_classes = ds.num_classes
        x = data.x  # torch.empty((data.x.shape[0], 0))
        ei = data.edge_index
        ea = torch.ones(ei.shape[1])
        y = data.y
        bg = BaseGraph(x, ei, ea, y)
        bg.num_classes = data.num_classes
        bg.y = bg.y.to(torch.int64)
        torch.save(bg, savepath)
        return bg

# def change(data):
#     x = data.graph['node_feat']
#     ei = data.graph['edge_index']
#     ea = data.graph['edge_feat']
#     y = data.label
#     bg = BaseGraph(x,ei,ea,y)
#     bg.num_classes=max(data.label.max().item() + 1, data.label.shape[1])
#     return bg