"""
PyTorch Tensors - The Foundation of Deep Learning
=================================================

Tensors are the fundamental data structure in PyTorch - multi-dimensional arrays
similar to NumPy's ndarrays but with GPU acceleration and automatic differentiation.
"""

import torch
import numpy as np

# =============================================================================
# TENSOR CREATION
# =============================================================================

# From Python data
tensor_from_list = torch.tensor([1, 2, 3, 4])
tensor_2d = torch.tensor([[1, 2], [3, 4]])

# Common initialization patterns
zeros = torch.zeros(3, 4)           # 3x4 tensor of zeros
ones = torch.ones(2, 3)             # 2x3 tensor of ones
empty = torch.empty(2, 2)           # uninitialized 2x2 tensor
rand = torch.rand(3, 3)             # uniform random [0, 1)
randn = torch.randn(3, 3)           # normal distribution (mean=0, std=1)
arange = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # 5 evenly spaced values from 0 to 1

# Identity and diagonal matrices
eye = torch.eye(3)                  # 3x3 identity matrix
diag = torch.diag(torch.tensor([1, 2, 3]))  # diagonal matrix

# Like operations (create tensor with same shape/dtype as another)
x = torch.randn(2, 3)
zeros_like_x = torch.zeros_like(x)
ones_like_x = torch.ones_like(x)
rand_like_x = torch.rand_like(x)

# =============================================================================
# DATA TYPES (dtype)
# =============================================================================

# PyTorch supports various data types
float32 = torch.tensor([1.0], dtype=torch.float32)  # default float type
float64 = torch.tensor([1.0], dtype=torch.float64)  # double precision
float16 = torch.tensor([1.0], dtype=torch.float16)  # half precision (GPU efficient)
bfloat16 = torch.tensor([1.0], dtype=torch.bfloat16)  # brain floating point

int32 = torch.tensor([1], dtype=torch.int32)
int64 = torch.tensor([1], dtype=torch.int64)  # default int type
bool_tensor = torch.tensor([True, False], dtype=torch.bool)

# Type conversion
x = torch.randn(3)
x_int = x.int()           # convert to int
x_long = x.long()         # convert to int64
x_float = x.float()       # convert to float32
x_double = x.double()     # convert to float64
x_half = x.half()         # convert to float16
x_to = x.to(torch.int16)  # explicit conversion

# =============================================================================
# TENSOR ATTRIBUTES
# =============================================================================

t = torch.randn(2, 3, 4)
print(f"Shape: {t.shape}")          # torch.Size([2, 3, 4])
print(f"Size: {t.size()}")          # same as shape
print(f"Dimensions: {t.dim()}")     # 3
print(f"Total elements: {t.numel()}")  # 24
print(f"Data type: {t.dtype}")      # torch.float32
print(f"Device: {t.device}")        # cpu or cuda:0

# =============================================================================
# DEVICE MANAGEMENT (CPU/GPU)
# =============================================================================

# Check CUDA availability
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Move tensors between devices
cpu_tensor = torch.randn(3, 3)

if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to('cuda')        # move to GPU
    gpu_tensor = cpu_tensor.cuda()            # equivalent
    back_to_cpu = gpu_tensor.to('cpu')        # move back to CPU
    back_to_cpu = gpu_tensor.cpu()            # equivalent

# Create tensor directly on GPU
if torch.cuda.is_available():
    gpu_direct = torch.randn(3, 3, device='cuda')

# Device-agnostic code pattern
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3, 3, device=device)

# =============================================================================
# INDEXING AND SLICING
# =============================================================================

t = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# Basic indexing
first_row = t[0]          # tensor([0, 1, 2, 3])
element = t[1, 2]         # tensor(6)
last_row = t[-1]          # tensor([8, 9, 10, 11])

# Slicing
first_two_rows = t[:2]    # rows 0 and 1
second_col = t[:, 1]      # all rows, column 1
submatrix = t[1:, 2:]     # rows 1+, columns 2+

# Advanced indexing
rows = torch.tensor([0, 2])
cols = torch.tensor([1, 3])
selected = t[rows, cols]  # elements at (0,1) and (2,3)

# Boolean indexing
mask = t > 5
filtered = t[mask]        # all elements > 5

# =============================================================================
# RESHAPING OPERATIONS
# =============================================================================

t = torch.arange(12)

# Reshape - returns a view when possible
reshaped = t.reshape(3, 4)
reshaped = t.reshape(3, -1)   # -1 infers dimension (4 in this case)

# View - always returns a view (tensor must be contiguous)
viewed = t.view(3, 4)

# Squeeze and unsqueeze
x = torch.randn(1, 3, 1, 4)
squeezed = x.squeeze()        # removes all dims of size 1: (3, 4)
squeezed_0 = x.squeeze(0)     # removes dim 0 if size 1: (3, 1, 4)

y = torch.randn(3, 4)
unsqueezed = y.unsqueeze(0)   # adds dim at position 0: (1, 3, 4)
unsqueezed = y.unsqueeze(-1)  # adds dim at end: (3, 4, 1)

# Flatten
flat = t.reshape(3, 4).flatten()           # flatten all dims
flat_partial = t.reshape(2, 2, 3).flatten(1)  # flatten from dim 1: (2, 6)

# Transpose and permute
t = torch.randn(2, 3, 4)
transposed = t.transpose(0, 2)  # swap dims 0 and 2: (4, 3, 2)
permuted = t.permute(2, 0, 1)   # reorder dims: (4, 2, 3)

# For 2D tensors
matrix = torch.randn(3, 4)
transposed_2d = matrix.T       # shorthand for transpose

# =============================================================================
# TENSOR OPERATIONS
# =============================================================================

a = torch.randn(3, 4)
b = torch.randn(3, 4)

# Element-wise arithmetic
add = a + b                   # or torch.add(a, b)
sub = a - b                   # or torch.sub(a, b)
mul = a * b                   # or torch.mul(a, b)
div = a / b                   # or torch.div(a, b)
pow_op = a ** 2               # or torch.pow(a, 2)

# In-place operations (modify tensor directly, end with _)
a.add_(1)                     # add 1 to all elements in-place
a.mul_(2)                     # multiply by 2 in-place

# Matrix multiplication
m1 = torch.randn(2, 3)
m2 = torch.randn(3, 4)
matmul = m1 @ m2              # or torch.matmul(m1, m2) -> (2, 4)
matmul = torch.mm(m1, m2)     # 2D only

# Batch matrix multiplication
batch1 = torch.randn(10, 2, 3)
batch2 = torch.randn(10, 3, 4)
batch_matmul = torch.bmm(batch1, batch2)  # (10, 2, 4)

# Einstein summation (flexible contraction)
einsum_matmul = torch.einsum('ij,jk->ik', m1, m2)

# =============================================================================
# REDUCTION OPERATIONS
# =============================================================================

t = torch.randn(3, 4)

# Global reductions
total_sum = t.sum()
mean_val = t.mean()
std_val = t.std()
var_val = t.var()
max_val = t.max()
min_val = t.min()
prod = t.prod()

# Reductions along dimensions
sum_rows = t.sum(dim=0)       # sum each column -> (4,)
sum_cols = t.sum(dim=1)       # sum each row -> (3,)
mean_cols = t.mean(dim=1, keepdim=True)  # keep dimension -> (3, 1)

# Argmax/Argmin (indices of max/min)
max_idx = t.argmax()          # global index of max
max_idx_dim = t.argmax(dim=1) # index of max in each row

# Max/Min with indices
max_vals, max_indices = t.max(dim=1)

# =============================================================================
# BROADCASTING
# =============================================================================

# PyTorch automatically broadcasts tensors of different shapes
a = torch.randn(3, 4)
b = torch.randn(4)        # broadcasts to (3, 4)
c = a + b                 # works!

a = torch.randn(3, 1)
b = torch.randn(1, 4)
c = a + b                 # broadcasts to (3, 4)

# Broadcasting rules:
# 1. Dimensions are compared from right to left
# 2. Dimensions are compatible if equal OR one of them is 1
# 3. Missing dimensions are treated as 1

# =============================================================================
# NUMPY INTEROPERABILITY
# =============================================================================

# NumPy to PyTorch
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)  # shares memory!
tensor_copy = torch.tensor(np_array) # creates copy

# PyTorch to NumPy
t = torch.randn(3, 3)
np_arr = t.numpy()        # shares memory (CPU only)
np_arr = t.detach().numpy()  # if tensor requires grad

# GPU tensors must be moved to CPU first
if torch.cuda.is_available():
    gpu_t = torch.randn(3, 3, device='cuda')
    np_arr = gpu_t.cpu().numpy()

# =============================================================================
# USEFUL TENSOR FUNCTIONS
# =============================================================================

t = torch.randn(3, 4)

# Clipping values
clipped = t.clamp(min=-1, max=1)  # values clamped to [-1, 1]
clipped_min = t.clamp(min=0)      # ReLU-like operation

# Absolute value
abs_t = t.abs()

# Sign
sign_t = t.sign()  # -1, 0, or 1

# Rounding
rounded = t.round()
floor_t = t.floor()
ceil_t = t.ceil()

# Comparison operations (return boolean tensors)
gt = t > 0
eq = t == 0
ne = t != 0

# Where (conditional selection)
condition = t > 0
selected = torch.where(condition, t, torch.zeros_like(t))  # ReLU

# Sorting
sorted_t, indices = t.sort(dim=1)

# Unique values
unique = torch.unique(t)

# Stacking and concatenation
t1, t2 = torch.randn(2, 3), torch.randn(2, 3)
stacked = torch.stack([t1, t2], dim=0)    # new dim: (2, 2, 3)
concat = torch.cat([t1, t2], dim=0)       # along existing dim: (4, 3)

# Split and chunk
chunks = t.chunk(2, dim=0)      # split into 2 chunks along dim 0
splits = t.split(2, dim=1)      # split into chunks of size 2

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Quick demo
    x = torch.randn(2, 3)
    print(f"\nRandom tensor:\n{x}")
    print(f"Shape: {x.shape}, dtype: {x.dtype}")
