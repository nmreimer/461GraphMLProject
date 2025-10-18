import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from sklearn.metrics import roc_auc_score

from torch_geometric.loader import DataLoader

from loadMoleculeData import load_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 100

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            g = global_mean_pool(x, batch)

        return self.lin(g).view(-1)

def do_epoch(loader, training: bool, model, optimizer = None, criterion = None):
    if training:
        model.train()
    else:
        model.eval()

    all_logits, all_labels, total_loss, n = [], [], 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        
        logits = logits.view(-1)
        labels = batch.y.view(-1).float() 
        loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    try:
        auc = roc_auc_score(all_labels, 1/(1+np.exp(-all_logits))) # TODO: Fix overflow
    except ValueError:
        auc = float("nan")
    return total_loss / n, auc

def train(datasets):
    for dataset in datasets:
        batch_size = 256
        train_loader = DataLoader(dataset['train_set'], batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(dataset['val_set'],   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(dataset['test_set'],  batch_size=batch_size, shuffle=False)

        # Get the original dataset to access num_features
        original_dataset = dataset['train_set'].dataset
        model = GCN(in_channels=original_dataset.num_features, hidden_channels=128, num_layers=3, dropout=0.2)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auc = 0.0

        for epoch in range(1, EPOCHS+1):
            train_loss, train_auc = do_epoch(train_loader, training=True, model=model, optimizer=optimizer, criterion=criterion)
            val_loss,   val_auc   = do_epoch(val_loader,   training=False, model=model, criterion=criterion)
            test_loss,  test_auc  = do_epoch(test_loader,  training=False, model=model, criterion=criterion)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), f"models/baselineGCN_{dataset['name']}.pt")

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | "
                    f"Train loss {train_loss:.4f} AUC {train_auc:.3f} | "
                    f"Val AUC {val_auc:.3f}")

        print(f"\nBest Val AUC: {best_val_auc:.3f}")
        ckpt = torch.load(f"models/baselineGCN_{dataset['name']}.pt", map_location=device)
        model.load_state_dict(ckpt)
        _, test_auc = do_epoch(test_loader, training=False, model=model, criterion=criterion)
        print(f"Test AUC (best-val checkpoint): {test_auc:.3f}")

def get_probs(model, dataset):
    test_loader = DataLoader(dataset['test_set'], batch_size=256, shuffle=False)
    model.eval()
    all_logits = []
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        all_logits.append(logits.detach().cpu())
    return torch.cat(all_logits).numpy()


if __name__ == "__main__":
    datasets = load_datasets()
    train(datasets)

