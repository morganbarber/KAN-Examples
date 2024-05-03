import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Spline activation function
class Spline(nn.Module):
    def __init__(self, num_intervals, order=3):
        super().__init__()
        self.num_intervals = num_intervals
        self.order = order
        self.knots = torch.linspace(0, 1, num_intervals + 1)
        self.basis = torch.cat([torch.zeros(order), torch.ones(1), torch.zeros(order)])
        self.coeffs = nn.Parameter(torch.randn(num_intervals + order))

    def forward(self, x):
        x = x.unsqueeze(-1)
        basis_matrix = torch.stack([
            torch.roll(self.basis, i).pow(self.order - 1)
            for i in range(self.num_intervals + self.order)
        ], dim=0)
        spline_values = torch.matmul(basis_matrix, self.coeffs)
        return torch.sum(spline_values * torch.prod(torch.relu(x - self.knots), dim=0), dim=1)

# KAN Layer
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_intervals, order=3):
        super().__init__()
        self.activations = nn.ModuleList([
            Spline(num_intervals, order) for _ in range(in_features * out_features)
        ])

    def forward(self, x):
        activations = torch.stack([f(x) for f in self.activations], dim=1)
        activations = activations.view(x.shape[0], -1, x.shape[1])
        return torch.sum(activations, dim=2)

# KAN Network
class KAN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_intervals, order=3):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLayer(input_dim, hidden_dims[0], num_intervals, order)
        ])
        for i in range(1, len(hidden_dims)):
            self.layers.append(KANLayer(hidden_dims[i-1], hidden_dims[i], num_intervals, order))
        self.layers.append(KANLayer(hidden_dims[-1], output_dim, num_intervals, order))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# MNIST Dataset and Data Loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Model, Loss Function, and Optimizer
model = KAN(784, [128, 64], 10, num_intervals=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training Function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Testing Function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.view(-1, 784))
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    for epoch in range(1, 10):
        train(epoch)
        test()