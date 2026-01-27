"""
PyTorch Autograd - Automatic Differentiation
=============================================

Autograd is PyTorch's automatic differentiation engine that powers neural network
training. It computes gradients automatically through a technique called reverse-mode
automatic differentiation (backpropagation).
"""

import torch

# =============================================================================
# BASICS OF AUTOGRAD
# =============================================================================

# requires_grad=True tells PyTorch to track operations for gradient computation
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0], requires_grad=True)

# Operations create a computational graph
z = x * y + x ** 2
out = z.sum()

print(f"x: {x}")
print(f"y: {y}")
print(f"z = x*y + x^2: {z}")
print(f"out = sum(z): {out}")

# Compute gradients via backpropagation
out.backward()

# Gradients are stored in .grad attribute
# d(out)/dx = y + 2x = [4+4, 5+6] = [8, 11]
print(f"\nGradient of out w.r.t. x: {x.grad}")
# d(out)/dy = x = [2, 3]
print(f"Gradient of out w.r.t. y: {y.grad}")

# =============================================================================
# THE COMPUTATIONAL GRAPH
# =============================================================================

"""
PyTorch builds a Dynamic Computational Graph (DCG):

1. Each tensor with requires_grad=True is a "leaf" node
2. Operations create "intermediate" nodes
3. The graph is built on-the-fly during forward pass
4. backward() traverses the graph in reverse to compute gradients
5. The graph is destroyed after backward() (unless retain_graph=True)

Example graph for z = x*y + x^2:

    x -----> (*) -----> (+) -----> z -----> sum -----> out
      \       ^          ^
       \      |          |
        `-> (^2)--------'
             |
    y ------'
"""

# =============================================================================
# GRADIENT ATTRIBUTES
# =============================================================================

a = torch.randn(3, requires_grad=True)
b = a * 2

print(f"\na.requires_grad: {a.requires_grad}")  # True (leaf)
print(f"b.requires_grad: {b.requires_grad}")    # True (result of operation)
print(f"a.is_leaf: {a.is_leaf}")                # True
print(f"b.is_leaf: {b.is_leaf}")                # False
print(f"b.grad_fn: {b.grad_fn}")                # MulBackward0

# Only leaf tensors retain gradients by default
c = b.sum()
c.backward()
print(f"a.grad: {a.grad}")    # [2, 2, 2]
print(f"b.grad: {b.grad}")    # None (non-leaf)

# To retain gradients for non-leaf tensors
a = torch.randn(3, requires_grad=True)
b = a * 2
b.retain_grad()  # explicitly retain
c = b.sum()
c.backward()
print(f"b.grad (retained): {b.grad}")

# =============================================================================
# GRADIENT COMPUTATION PATTERNS
# =============================================================================

# Pattern 1: Simple scalar output
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(f"\ndy/dx where y = sum(x^2): {x.grad}")  # [2, 4, 6]

# Pattern 2: Gradient of vector output requires a vector argument
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # vector output

# Must provide gradient argument (usually ones for each output element)
# This is like computing sum(v * dy/dx) where v is the gradient argument
v = torch.ones_like(y)
y.backward(gradient=v)
print(f"dy/dx with gradient=[1,1,1]: {x.grad}")

# Pattern 3: Jacobian-vector product (JVP)
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
v = torch.tensor([0.1, 1.0, 0.01])
y.backward(gradient=v)
print(f"Jacobian-vector product: {x.grad}")  # [0.2, 4.0, 0.06]

# =============================================================================
# CONTROLLING GRADIENT COMPUTATION
# =============================================================================

# torch.no_grad() - disable gradient tracking (inference mode)
x = torch.randn(3, requires_grad=True)

with torch.no_grad():
    y = x * 2
    print(f"\nIn no_grad: y.requires_grad = {y.requires_grad}")  # False

# torch.enable_grad() - enable gradient tracking (nested in no_grad)
with torch.no_grad():
    with torch.enable_grad():
        y = x * 2
        print(f"In enable_grad: y.requires_grad = {y.requires_grad}")  # True

# torch.inference_mode() - faster than no_grad, but more restrictive
with torch.inference_mode():
    y = x * 2
    # Can't use y for anything requiring gradients later

# .detach() - returns a tensor detached from the graph
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.detach()  # z shares data but doesn't track gradients
print(f"z.requires_grad: {z.requires_grad}")  # False

# =============================================================================
# GRADIENT ACCUMULATION
# =============================================================================

# Gradients accumulate by default!
x = torch.tensor([1.0, 2.0], requires_grad=True)

for i in range(3):
    y = (x ** 2).sum()
    y.backward()
    print(f"Iteration {i}: x.grad = {x.grad}")

# Output shows accumulation:
# Iteration 0: x.grad = [2, 4]
# Iteration 1: x.grad = [4, 8]
# Iteration 2: x.grad = [6, 12]

# IMPORTANT: Zero gradients before each backward pass
x = torch.tensor([1.0, 2.0], requires_grad=True)

for i in range(3):
    if x.grad is not None:
        x.grad.zero_()  # zero the gradients
    y = (x ** 2).sum()
    y.backward()
    print(f"Iteration {i} (zeroed): x.grad = {x.grad}")

# =============================================================================
# HIGHER-ORDER GRADIENTS
# =============================================================================

x = torch.tensor([2.0], requires_grad=True)

# First derivative: y = x^3, dy/dx = 3x^2
y = x ** 3
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"\nFirst derivative of x^3 at x=2: {grad1}")  # 12.0

# Second derivative: d2y/dx2 = 6x
grad2 = torch.autograd.grad(grad1, x, create_graph=True)[0]
print(f"Second derivative of x^3 at x=2: {grad2}")  # 12.0

# Third derivative: d3y/dx3 = 6
grad3 = torch.autograd.grad(grad2, x)[0]
print(f"Third derivative of x^3 at x=2: {grad3}")  # 6.0

# =============================================================================
# torch.autograd.grad() vs .backward()
# =============================================================================

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()

# .backward() stores gradients in .grad attributes
# y.backward()
# grad = x.grad

# torch.autograd.grad() returns gradients directly
grad = torch.autograd.grad(y, x)[0]
print(f"\nGradient via autograd.grad: {grad}")

# Useful for:
# 1. Computing gradients without modifying .grad
# 2. Computing gradients w.r.t. non-leaf tensors
# 3. Higher-order gradients (with create_graph=True)

# =============================================================================
# GRADIENT CHECKPOINTING (Memory Optimization)
# =============================================================================

from torch.utils.checkpoint import checkpoint

def compute_heavy(x):
    """Simulates a memory-intensive computation."""
    y = x ** 2
    z = y ** 2
    return z ** 2

x = torch.randn(100, requires_grad=True)

# Normal forward pass: stores all intermediate activations
y_normal = compute_heavy(x)

# Checkpointed: trades compute for memory
# Recomputes forward pass during backward to save memory
y_checkpoint = checkpoint(compute_heavy, x, use_reentrant=False)

# =============================================================================
# CUSTOM AUTOGRAD FUNCTIONS
# =============================================================================

class MyReLU(torch.autograd.Function):
    """Custom ReLU implementation with explicit forward and backward."""

    @staticmethod
    def forward(ctx, input):
        # ctx is a context object to save tensors for backward
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is the gradient from the next layer
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Use the custom function
my_relu = MyReLU.apply

x = torch.randn(5, requires_grad=True)
y = my_relu(x)
y.sum().backward()
print(f"\nCustom ReLU gradient: {x.grad}")

# =============================================================================
# COMMON PITFALLS AND SOLUTIONS
# =============================================================================

# Pitfall 1: In-place operations can break the graph
x = torch.randn(3, requires_grad=True)
y = x * 2
# y.add_(1)  # This would cause an error during backward!
# Solution: Use out-of-place operations
y = y + 1  # Creates new tensor instead

# Pitfall 2: Modifying tensor data directly
x = torch.randn(3, requires_grad=True)
# x.data[0] = 0  # Bypasses autograd, gradients may be wrong
# Solution: Create new tensor with modifications

# Pitfall 3: Forgetting to zero gradients
# Always zero gradients before backward (see Gradient Accumulation section)

# Pitfall 4: Using Python scalars instead of tensors
# x = torch.tensor(2.0, requires_grad=True)
# y = x * 3  # OK
# z = x * 3.0  # Also OK, PyTorch handles this
# But be careful with complex expressions

# =============================================================================
# DEBUGGING AUTOGRAD
# =============================================================================

# Check if a tensor requires gradients
x = torch.randn(3, requires_grad=True)
print(f"\nx.requires_grad: {x.requires_grad}")

# Check the gradient function
y = x * 2
print(f"y.grad_fn: {y.grad_fn}")

# Enable anomaly detection (slower but helpful for debugging)
# torch.autograd.set_detect_anomaly(True)

# Print the computational graph (for debugging)
def print_graph(tensor, indent=0):
    """Recursively print the computational graph."""
    print("  " * indent + str(tensor.grad_fn))
    if tensor.grad_fn:
        for t in tensor.grad_fn.next_functions:
            if t[0]:
                # t[0] is the grad_fn, t[1] is the output index
                print_graph(type('', (), {'grad_fn': t[0]})(), indent + 1)

x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
z = y * 3
print("\nComputational graph for z = 3 * x^2:")
print_graph(z)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Autograd Demo Complete!")
    print("="*50)

    # Simple gradient descent example
    x = torch.tensor([5.0], requires_grad=True)
    learning_rate = 0.1

    print("\nSimple gradient descent to minimize x^2:")
    for i in range(10):
        y = x ** 2
        y.backward()

        with torch.no_grad():
            x -= learning_rate * x.grad

        x.grad.zero_()
        print(f"  Step {i+1}: x = {x.item():.4f}, x^2 = {(x**2).item():.4f}")
