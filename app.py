from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv

data = Planetoid(root  = '.', name = 'cora', transform = NormalizeFeatures())
data_ext = data[0]

#Intializing GNN Layers
class GNNClassification(nn.Module):
  def __init__(self, channels):
    super(GNNClassification, self).__init__()
    torch.manual_seed(42)
    #intializing layers
    self.GNNLayer1 = GCNConv(data.num_features, channels)
    self.GNNLayer2 = GCNConv(channels, channels)
    self.GNNLayer3 = GCNConv(channels, channels)
    self.GNNLayer4 = GCNConv(channels, channels)
    #output layer
    self.GNNOut = Linear(channels, data.num_classes)

  def forward(self, x, edge_index):
    x = self.GNNLayer1(x, edge_index)
    x = x.relu()
    x = F.dropout(x, p = 0.2, training = self.training)

    x = self.GNNLayer2(x, edge_index)
    x = x.relu()
    x = F.dropout(x, p = 0.2, training = self.training)

    x = self.GNNLayer3(x, edge_index)
    x = x.relu()
    x = F.dropout(x, p = 0.2, training = self.training)

    x = self.GNNLayer4(x, edge_index)
    x = x.relu()
    x = F.dropout(x, p = 0.2, training = self.training)

    y = F.softmax(self.GNNOut(x), dim = 1)
    return y

model = GNNClassification(channels = 16)

#Training model

learning_rate = 3e-4

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)



def train():
  model.train()
  optimizer.zero_grad()
  out = model(data_ext.x, data_ext.edge_index)

  loss = F.cross_entropy(out[data_ext.train_mask], data.y[data_ext.train_mask])
  loss.backward()
  optimizer.step()
  return loss

losses = []
for epoch in range(12000):
  loss = train()
  losses.append(loss)
  if epoch%200 == 0:
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

#testing and evaluation
def test():
  model.eval()
  out = model(data_ext.x, data_ext.edge_index)

  pred = out.argmax(dim=1)

  test_correct  = pred[data.test_mask] == data.y[data.test_mask]

  test_acc = int(test_correct.sum())/int(data.test_mask.sum())


  return test_acc

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')