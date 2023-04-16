import os
import numpy as np
from PIL import Image
import torch
import torch_cluster
import torch_scatter
import torch_sparse
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F


class GNN(MessagePassing):
    def __init__(self):
        super(GNN, self).__init__(aggr='add')  # Use "add" aggregation for messages

        self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=1)  # Convolutional layer for node features
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=1)  # Convolutional layer for node features
        self.fc1 = torch.nn.Linear(64, 128)  # Linear layer for node features
        self.fc2 = torch.nn.Linear(128, 26)  # Linear layer for output

    def forward(self, x, edge_index):
        # x is the node feature matrix
        # edge_index is the tensor of node indices specifying the edge connections

        # Send messages between nodes
        x = self.conv1(x)
        x = self.conv2(x)
        self.propagate(edge_index, x=x)

        # Update node features based on messages received
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def message(self, x_j):
        # Aggregate messages from adjacent nodes
        return x_j
    

# Load data from directory of character images and labels
data_dir = './chars/'
data_list = []
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    for filename in os.listdir(label_dir):
        # Open image, convert to grayscale, and resize to 100x100
        img = Image.open(os.path.join(label_dir, filename)).convert('L')
        img = img.resize((100, 100))
        # Convert image pixel values to torch tensor and reshape into a 2D matrix
        img_tensor = torch.tensor(np.array(img)).reshape((1, 100*100)).float()
        # Create edge_index tensor to represent edges between adjacent pixels (assuming 8-connectivity)
        n = img_tensor.shape[1]
        row = []
        col = []
        for i in range(n):
            r = i//100
            c = i%100
            if r > 0:
                row += [i, i]
                col += [i-100, i]
            if c > 0:
                row += [i, i]
                col += [i-1, i]
            if r < 99:
                row += [i, i]
                col += [i+100, i]
            if c < 99:
                row += [i, i]
                col += [i+1, i]
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_index = edge_index.to('cuda')
        # Create a Data object to represent the graph
        data = Data(x=img_tensor, edge_index=edge_index)
        data_list.append((data, label))

# Define a DataLoader to batch the data
batch_size = 32
loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

# Initialize and train the model
model = GNN().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    total_loss = 0.
    for batch_data, batch_labels in loader:
        batch_data, batch_labels = batch_data.to('cuda'), batch_labels.to('cuda')
        optimizer.zero_grad()
        out = model(batch_data.x, batch_data.edge_index)
        loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_data.num_graphs
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_list)}')