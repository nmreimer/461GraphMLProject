import numpy as np
from torch.utils.data import Subset
from torch_geometric.datasets import MoleculeNet

from deepchem.splits import ScaffoldSplitter
from deepchem.data import NumpyDataset

import pandas as pd

def load_datasets(names = ["BBBP", "BACE", "HIV"], root = "data/moleculenet"): # default is BBBP, BACE, and HIV which are single task classification datasets
    datasets = []
    for name in names:
        dataset = MoleculeNet(root=root, name=name)
        ids = [d.smiles for d in dataset]
        y = None
        X = ids  
        dc_ds = NumpyDataset(X=X, y=y, ids=ids)
        splitter = ScaffoldSplitter()   # Bemis–Murcko scaffold splitter
        train_idx, valid_idx, test_idx = splitter.split(dc_ds, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42)
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, valid_idx)
        test_set = Subset(dataset, test_idx)
        datasets.append({
            'name': name,
            'train_set': train_set,
            'val_set': val_set,
            'test_set': test_set
        })
    return datasets



if __name__ == "__main__":
    datasets = load_datasets()
    print(datasets)