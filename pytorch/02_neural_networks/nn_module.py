"""
PyTorch Neural Networks - nn.Module and Building Blocks
========================================================

The nn.Module class is the base for all neural networks.
It organizes parameters, sub-modules, and forward logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ===========================================================================
# BASIC nn.Module STRUCTURE
# ===========================================================================


class SimpleNetwork(nn.Module):
    """A simple feedforward neural network."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # Always call parent constructor first

        # Define layers as attributes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Non-parameterized layers can also be defined
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Define the forward pass."""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Usage
model = SimpleNetwork(784, 256, 10)
x = torch.randn(32, 784)  # batch of 32, 784 features each
output = model(x)  # calls forward() automatically
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# ===========================================================================
# COMMON LAYER TYPES
# ===========================================================================

# --- Linear (Fully Connected) Layers ---
linear = nn.Linear(in_features=128, out_features=64)
linear_no_bias = nn.Linear(128, 64, bias=False)

# --- Convolutional Layers ---
# Conv2d(in_channels, out_channels, kernel_size, stride, padding)
conv2d = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
conv1d = nn.Conv1d(1, 32, kernel_size=5)

# Transposed convolution (upsampling)
conv_transpose = nn.ConvTranspose2d(
    64, 32, kernel_size=4, stride=2, padding=1
)

# --- Pooling Layers ---
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2)
adaptive_pool = nn.AdaptiveAvgPool2d(
    (1, 1)
)  # output size (1, 1) regardless of input

# --- Normalization Layers ---
batch_norm_2d = nn.BatchNorm2d(64)  # for conv layers
batch_norm_1d = nn.BatchNorm1d(256)  # for linear layers
layer_norm = nn.LayerNorm(256)  # normalizes over last dimension
group_norm = nn.GroupNorm(8, 64)  # 8 groups, 64 channels

# --- Dropout Layers ---
dropout = nn.Dropout(p=0.5)  # standard dropout
dropout2d = nn.Dropout2d(p=0.5)  # for conv layers (drops entire channels)

# --- Activation Functions ---
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
gelu = nn.GELU()  # used in transformers
silu = nn.SiLU()  # swish activation
softmax = nn.Softmax(dim=-1)
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

# --- Recurrent Layers ---
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
lstm = nn.LSTM(
    input_size=128, hidden_size=256, num_layers=2, batch_first=True
)
gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=True)

# --- Embedding ---
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=256)

# ===========================================================================
# nn.Sequential - QUICK MODEL BUILDING
# ===========================================================================

# For simple sequential architectures
sequential_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 10),
)

# With named modules (for easier access)
from collections import OrderedDict

named_sequential = nn.Sequential(
    OrderedDict(
        [
            ("fc1", nn.Linear(784, 256)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(256, 10)),
        ]
    )
)

# Access layers
print(f"First layer: {named_sequential.fc1}")

# ===========================================================================
# nn.ModuleList and nn.ModuleDict
# ===========================================================================


class DynamicNetwork(nn.Module):
    """Network with dynamic number of layers."""

    def __init__(self, layer_sizes):
        super().__init__()

        # ModuleList: use when layers are indexed numerically
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )

        # ModuleDict: use when layers are accessed by name
        self.activations = nn.ModuleDict(
            {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}
        )

    def forward(self, x, activation="relu"):
        for layer in self.layers[:-1]:
            x = self.activations[activation](layer(x))
        return self.layers[-1](x)


# Why use ModuleList/ModuleDict instead of regular list/dict?
# Regular Python containers don't register sub-modules properly:
# - Parameters won't be included in model.parameters()
# - model.to(device) won't move them
# - state_dict won't include them

# ===========================================================================
# PARAMETER ACCESS AND MANIPULATION
# ===========================================================================

model = SimpleNetwork(784, 256, 10)

# Iterate over all parameters
print("\nAll parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

# Get total number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Freeze specific layers
for param in model.fc1.parameters():
    param.requires_grad = False

# Access specific layer's parameters
print(f"\nfc1 weight shape: {model.fc1.weight.shape}")
print(f"fc1 bias shape: {model.fc1.bias.shape}")

# ===========================================================================
# CUSTOM PARAMETER INITIALIZATION
# ===========================================================================


def init_weights(module):
    """Custom weight initialization function."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight, mode="fan_out", nonlinearity="relu"
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# Apply to model
model = SimpleNetwork(784, 256, 10)
model.apply(init_weights)

# Common initialization methods
# nn.init.xavier_uniform_(tensor)      # Good for tanh/sigmoid
# nn.init.xavier_normal_(tensor)
# nn.init.kaiming_uniform_(tensor)     # Good for ReLU
# nn.init.kaiming_normal_(tensor)
# nn.init.zeros_(tensor)
# nn.init.ones_(tensor)
# nn.init.constant_(tensor, value)
# nn.init.normal_(tensor, mean, std)
# nn.init.uniform_(tensor, a, b)

# ===========================================================================
# MODEL STATE DICT (Saving and Loading)
# ===========================================================================

model = SimpleNetwork(784, 256, 10)

# Save only model parameters (recommended)
torch.save(model.state_dict(), "model_weights.pth")

# Load parameters
loaded_model = SimpleNetwork(784, 256, 10)
loaded_model.load_state_dict(
    torch.load("model_weights.pth", weights_only=True)
)

# Save entire model (not recommended - pickle-based, less portable)
# torch.save(model, 'full_model.pth')
# loaded = torch.load('full_model.pth')

# Inspect state dict
state_dict = model.state_dict()
print("\nState dict keys:")
for key in state_dict.keys():
    print(f"  {key}")

# Clean up
import os

os.remove("model_weights.pth")

# ===========================================================================
# FUNCTIONAL API (torch.nn.functional)
# ===========================================================================


class FunctionalNetwork(nn.Module):
    """Using functional API for activations and operations."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        # No need to define activations - use functional

    def forward(self, x):
        # Use F.relu instead of nn.ReLU()
        x = F.relu(self.fc1(x))
        x = F.dropout(
            x, p=0.5, training=self.training
        )  # Note: training flag
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


# Functional is useful for:
# - Operations that don't have learnable parameters
# - When you need more control (e.g., different dropout rates)
# - Avoiding creating module objects for simple operations

# ===========================================================================
# CONVOLUTIONAL NEURAL NETWORK EXAMPLE
# ===========================================================================


class ConvNet(nn.Module):
    """CNN for image classification."""

    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Test CNN
cnn = ConvNet(num_classes=10)
images = torch.randn(4, 3, 32, 32)  # batch of 4, 3 channels, 32x32
output = cnn(images)
print(f"\nCNN input: {images.shape}")
print(f"CNN output: {output.shape}")

# ===========================================================================
# TRAINING AND EVALUATION MODES
# ===========================================================================

model = SimpleNetwork(784, 256, 10)

# Training mode - enables dropout, batch norm uses batch statistics
model.train()
print(f"\nTraining mode: {model.training}")

# Evaluation mode - disables dropout, batch norm uses running statistics
model.eval()
print(f"Evaluation mode (training=False): {model.training}")

# Proper inference pattern
model.eval()
with torch.no_grad():  # Disable gradient computation for efficiency
    test_input = torch.randn(1, 784)
    prediction = model(test_input)

# ===========================================================================
# DEVICE MANAGEMENT
# ===========================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Move model to device
model = SimpleNetwork(784, 256, 10).to(device)

# Move data to same device
x = torch.randn(32, 784).to(device)
output = model(x)

# Check where model parameters are
print(f"Model is on: {next(model.parameters()).device}")

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("nn.Module Demo Complete!")
    print("=" * 50)
