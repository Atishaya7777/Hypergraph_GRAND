import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from data.dataset import create_hypergraph_dataset

def dataset_split(split_type):
    print("="* 30, split_type + " SPLIT", "="*30)
    dataset = Planetoid(root='/tmp/Cora_standard', name='Cora', transform=NormalizeFeatures(), split="public")
    data = dataset[0]

    data.get('train_mask')

    c = 0

    for item in data.train_mask:
        if item == True:
            c += 1

    print(f"Train mask: {c/len(data.train_mask)}")

    c = 0

    for item in data.val_mask:
        if item == True:
            c += 1

    print(f"Validation mask: {c/len(data.val_mask)}")

    c = 0

    for item in data.test_mask:
        if item == True:
            c += 1

    print(f"Test mask: {c/len(data.test_mask)}")


dataset_split("public")
