import torch
from torch import nn
from torch.utils.data import DataLoader , TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from handle_data import Data

if torch.cuda.is_available():
    device = torch.device("cuda")
else :
    device = torch.device("cpu")

print(f'Using device {device}')

# simple shallow Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, activation_func): # Pass activation here
        super().__init__()
        # 1. Define the layers inside __init__ so PyTorch can "see" them
        data = Data()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(data.get_input_size(), 512),
            activation_func,
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # 2. Don't call a setup function here; just use the layers
        logits = self.linear_relu_stack(x)
        return logits

learning_rate = 1e-3
batch_size = 32
epochs = 70
# Pass the activation function when creating the model
model = NeuralNetwork(activation_func=nn.ReLU())
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


lst_acc = []
lst_lst = []
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    lst_acc.append(correct)
    lst_lst.append(test_loss)

data_obj = Data()
x_train, x_test, y_train, y_test = data_obj.handleXY()

# 1. Convert to Tensors
train_ds = TensorDataset(
    torch.tensor(x_train, dtype=torch.float32),
    torch.tensor(y_train.values, dtype=torch.long)
)
test_ds = TensorDataset(
    torch.tensor(x_test, dtype=torch.float32),
    torch.tensor(y_test.values, dtype=torch.long)
)

# 2. Create DataLoaders
train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=64)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print(lst_acc)
print(lst_lst)

#  Save Model
torch.save(model.state_dict(), "smoke_model.pt")
print("Saved PyTorch Model State to smoke_model.pt")


