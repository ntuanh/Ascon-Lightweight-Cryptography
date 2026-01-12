import torch
from torch import nn
from model.handle_data import Data

# -----------------------------
# Model definition (EXACT MATCH)
# -----------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, activation_func):
        super().__init__()
        data = Data()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(data.get_input_size(), 16),
            activation_func,
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

# -----------------------------
# Load trained model
# -----------------------------
model = NeuralNetwork(nn.ReLU())
model.load_state_dict(torch.load("smoke_model.pt", map_location="cpu"))
model.eval()

print("Model loaded successfully")

# -----------------------------
# C++ export helpers
# -----------------------------
def save_matrix_cpp(f, name, mat):
    rows, cols = mat.shape
    f.write(f"float {name}[{rows}][{cols}] = {{\n")
    for r in range(rows):
        row = ", ".join(f"{mat[r][c]:.8f}" for c in range(cols))
        f.write(f"  {{{row}}},\n")
    f.write("};\n\n")

def save_vector_cpp(f, name, vec):
    f.write(f"float {name}[{len(vec)}] = {{\n  ")
    f.write(", ".join(f"{v:.8f}" for v in vec))
    f.write("\n};\n\n")

# -----------------------------
# Export ALL Linear layers
# -----------------------------
linear_layer_indices = [0, 2, 4, 6]

with open("nn_weights.h", "w") as f:
    f.write("// AUTO-GENERATED FILE — DO NOT EDIT\n\n")

    layer_id = 0
    for idx in linear_layer_indices:
        layer = model.linear_relu_stack[idx]

        W = layer.weight.detach().cpu().numpy()
        b = layer.bias.detach().cpu().numpy()

        save_matrix_cpp(f, f"W{layer_id}", W)
        save_vector_cpp(f, f"b{layer_id}", b)

        print(f"Exported Layer {layer_id}: W{layer_id}{W.shape}, b{layer_id}{b.shape}")
        layer_id += 1

print("All layers exported to nn_weights.h")
