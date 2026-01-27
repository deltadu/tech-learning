# Practical Linear Algebra

Applications in machine learning, computer graphics, and robotics.

---

## Table of Contents

1. [NumPy Essentials](#numpy-essentials)
2. [Machine Learning Applications](#machine-learning-applications)
3. [Computer Graphics](#computer-graphics)
4. [Robotics Applications](#robotics-applications)
5. [Numerical Considerations](#numerical-considerations)
6. [Common Patterns](#common-patterns)

---

## NumPy Essentials

### Creating Arrays

```python
import numpy as np

# Vectors
v = np.array([1, 2, 3])
v = np.zeros(3)
v = np.ones(3)
v = np.arange(0, 10, 2)        # [0, 2, 4, 6, 8]
v = np.linspace(0, 1, 5)       # 5 evenly spaced

# Matrices
A = np.array([[1, 2], [3, 4]])
A = np.zeros((3, 4))           # 3×4 zeros
A = np.ones((3, 4))
A = np.eye(3)                  # 3×3 identity
A = np.diag([1, 2, 3])         # diagonal matrix
A = np.random.randn(3, 4)      # random normal
```

### Basic Operations

```python
# Vector operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b                # element-wise add
a * b                # element-wise multiply (Hadamard)
np.dot(a, b)         # dot product: 32
a @ b                # dot product (Python 3.5+): 32
np.cross(a, b)       # cross product (3D)

np.linalg.norm(a)    # L2 norm
np.linalg.norm(a, 1) # L1 norm
a / np.linalg.norm(a) # normalize

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A + B                # element-wise add
A * B                # element-wise multiply
A @ B                # matrix multiply
np.dot(A, B)         # matrix multiply
A.T                  # transpose

np.linalg.inv(A)     # inverse
np.linalg.det(A)     # determinant
np.trace(A)          # trace
np.linalg.matrix_rank(A)  # rank
```

### Solving Linear Systems

```python
# Solve Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 5])

x = np.linalg.solve(A, b)  # x = [2, 1]

# Least squares (for overdetermined systems)
A = np.array([[1, 1], [1, 2], [1, 3]])
b = np.array([1, 2, 2])
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
```

### Eigenvalues and SVD

```python
A = np.array([[4, 1], [2, 3]])

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvalues: [5, 2]
# eigenvectors: columns are eigenvectors

# For symmetric matrices (faster, numerically stable)
eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)

# SVD
U, s, Vt = np.linalg.svd(A)
# s is 1D array of singular values
# Reconstruct: A = U @ np.diag(s) @ Vt

# Truncated SVD (keep top k)
k = 2
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

### Broadcasting

NumPy automatically expands dimensions:

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2×3

v = np.array([10, 20, 30])  # 1D, length 3

A + v  # v broadcasts to each row
# [[11, 22, 33],
#  [14, 25, 36]]

# Subtract mean from each column
A - A.mean(axis=0)

# Normalize each row
A / np.linalg.norm(A, axis=1, keepdims=True)
```

---

## Machine Learning Applications

### Linear Regression

```
y = Xw + b

Find weights w that minimize ‖y - Xw‖²
```

**Closed-form solution (Normal Equation):**
```python
# Add bias column
X_b = np.hstack([np.ones((X.shape[0], 1)), X])

# Normal equation: w = (XᵀX)⁻¹Xᵀy
w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Or more numerically stable:
w = np.linalg.lstsq(X_b, y, rcond=None)[0]

# Or using pseudoinverse:
w = np.linalg.pinv(X_b) @ y
```

### Principal Component Analysis (PCA)

Reduce dimensionality by projecting onto top eigenvectors of covariance matrix.

```python
def pca(X, n_components):
    # Center the data
    X_centered = X - X.mean(axis=0)

    # Covariance matrix
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project onto top components
    components = eigenvectors[:, :n_components]
    X_projected = X_centered @ components

    return X_projected, components

# Or using SVD (more numerically stable):
def pca_svd(X, n_components):
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T, Vt[:n_components]
```

### Cosine Similarity

Measure similarity between vectors (used in embeddings, recommendations):

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# For matrices (each row is a vector)
def pairwise_cosine(A, B):
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T
```

### Neural Network Forward Pass

Neural networks are sequences of matrix operations:

```python
def forward(X, W1, b1, W2, b2):
    # Layer 1: linear + ReLU
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)  # ReLU

    # Layer 2: linear + softmax
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    return A2

def softmax(x):
    exp_x = np.exp(x - x.max(axis=1, keepdims=True))  # numerical stability
    return exp_x / exp_x.sum(axis=1, keepdims=True)
```

### Batch Normalization

Normalize activations using mean and variance:

```python
def batch_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

---

## Computer Graphics

### 2D Transformations

All 2D transformations as matrices:

**Translation** (requires homogeneous coordinates):
```
┌ 1  0  tx ┐   ┌ x ┐   ┌ x + tx ┐
│ 0  1  ty │ × │ y │ = │ y + ty │
│ 0  0  1  │   │ 1 │   │   1    │
└          ┘   └   ┘   └        ┘
```

**Rotation** (counterclockwise by θ):
```
┌ cos θ  -sin θ  0 ┐
│ sin θ   cos θ  0 │
│   0       0    1 │
└                  ┘
```

**Scaling**:
```
┌ sx  0   0 ┐
│ 0   sy  0 │
│ 0   0   1 │
└           ┘
```

**Combining transformations:**
```python
def transform_2d(points, transformations):
    # points: Nx2 array
    # Add homogeneous coordinate
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])

    # Combine all transformations (right to left)
    M = np.eye(3)
    for T in transformations:
        M = T @ M

    # Apply transformation
    result = (M @ points_h.T).T
    return result[:, :2]

# Example: rotate then translate
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta),  np.cos(theta), 0],
              [0, 0, 1]])

T = np.array([[1, 0, 10],
              [0, 1, 5],
              [0, 0, 1]])

combined = T @ R  # first rotate, then translate
```

### 3D Transformations

**Rotation around Z-axis:**
```
┌ cos θ  -sin θ  0  0 ┐
│ sin θ   cos θ  0  0 │
│   0       0    1  0 │
│   0       0    0  1 │
└                     ┘
```

**Rotation around X-axis:**
```
┌ 1    0       0    0 ┐
│ 0  cos θ  -sin θ  0 │
│ 0  sin θ   cos θ  0 │
│ 0    0       0    1 │
└                     ┘
```

**Rotation around Y-axis:**
```
┌  cos θ  0  sin θ  0 ┐
│    0    1    0    0 │
│ -sin θ  0  cos θ  0 │
│    0    0    0    1 │
└                     ┘
```

### Perspective Projection

Project 3D to 2D (camera):

```
┌ f/aspect  0    0       0    ┐
│    0      f    0       0    │
│    0      0  (f+n)/(n-f)  (2fn)/(n-f) │
│    0      0   -1       0    │
└                             ┘

f = focal length (cot(fov/2))
n = near plane
far = far plane
aspect = width/height
```

### Quaternions for 3D Rotation

Quaternions avoid gimbal lock and interpolate smoothly:

```python
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def axis_angle_to_quaternion(axis, angle):
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    return np.array([
        np.cos(half_angle),
        axis[0] * np.sin(half_angle),
        axis[1] * np.sin(half_angle),
        axis[2] * np.sin(half_angle)
    ])
```

---

## Robotics Applications

### Homogeneous Transformations

Combine rotation and translation in one matrix:

```
┌ R  t ┐     R = 3×3 rotation matrix
│ 0  1 │     t = 3×1 translation vector
└      ┘

This is a 4×4 matrix that transforms homogeneous coordinates.
```

```python
def homogeneous_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def transform_points(T, points):
    # points: Nx3
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = (T @ points_h.T).T
    return transformed[:, :3]
```

### Forward Kinematics

Chain of transformations from base to end effector:

```
T_base_to_end = T_0_1 @ T_1_2 @ T_2_3 @ ... @ T_n-1_n

Each T_i is the transform from joint i to joint i+1
```

```python
def forward_kinematics(joint_angles, dh_params):
    """
    DH parameters: list of (a, alpha, d, theta_offset)
    joint_angles: current joint angles
    """
    T = np.eye(4)
    for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
        theta = joint_angles[i] + theta_offset
        T_i = dh_transform(a, alpha, d, theta)
        T = T @ T_i
    return T

def dh_transform(a, alpha, d, theta):
    """Denavit-Hartenberg transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])
```

### Jacobian for Inverse Kinematics

Relate joint velocities to end-effector velocities:

```
ẋ = J(q) × q̇

x = end-effector pose
q = joint angles
J = Jacobian matrix (6×n for n joints)
```

```python
def numerical_jacobian(forward_fn, q, epsilon=1e-6):
    """Compute Jacobian numerically."""
    n_joints = len(q)
    x0 = forward_fn(q)  # current end-effector pose

    J = np.zeros((len(x0), n_joints))
    for i in range(n_joints):
        q_plus = q.copy()
        q_plus[i] += epsilon
        x_plus = forward_fn(q_plus)
        J[:, i] = (x_plus - x0) / epsilon

    return J

def inverse_kinematics_step(current_q, target_x, forward_fn, learning_rate=0.1):
    """One step of iterative IK using Jacobian pseudoinverse."""
    current_x = forward_fn(current_q)
    error = target_x - current_x

    J = numerical_jacobian(forward_fn, current_q)
    J_pinv = np.linalg.pinv(J)

    delta_q = J_pinv @ error
    new_q = current_q + learning_rate * delta_q

    return new_q
```

### Kalman Filter

Optimal state estimation using linear algebra:

```python
def kalman_filter(x, P, z, F, H, Q, R):
    """
    x: state estimate
    P: state covariance
    z: measurement
    F: state transition matrix
    H: observation matrix
    Q: process noise covariance
    R: measurement noise covariance
    """
    # Predict
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    # Update
    y = z - H @ x_pred                          # innovation
    S = H @ P_pred @ H.T + R                    # innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)         # Kalman gain

    x_new = x_pred + K @ y
    P_new = (np.eye(len(x)) - K @ H) @ P_pred

    return x_new, P_new
```

### Least Squares for Sensor Fusion

Combine multiple sensor readings:

```python
def weighted_least_squares(measurements, jacobians, covariances):
    """
    Combine multiple measurements with different noise levels.

    measurements: list of measurement vectors
    jacobians: list of Jacobian matrices (how state affects each measurement)
    covariances: list of measurement covariance matrices
    """
    # Stack everything
    H = np.vstack(jacobians)
    z = np.hstack(measurements)

    # Weight matrix (inverse of covariance)
    W = np.linalg.inv(block_diag(*covariances))

    # Weighted least squares: x = (H'WH)^-1 H'Wz
    HtW = H.T @ W
    x = np.linalg.solve(HtW @ H, HtW @ z)

    return x
```

---

## Numerical Considerations

### Condition Number

Measures how sensitive a solution is to input errors:

```python
cond = np.linalg.cond(A)

# cond ≈ 1: well-conditioned
# cond > 10^10: ill-conditioned (results may be unreliable)
```

### Avoid Explicit Inverse

```python
# ❌ Bad: explicitly computing inverse
x = np.linalg.inv(A) @ b

# ✓ Good: solve the system directly
x = np.linalg.solve(A, b)
```

### Numerical Stability Tips

```python
# Use stable algorithms
np.linalg.lstsq(A, b)  # instead of inv(A.T @ A) @ A.T @ b
np.linalg.eigh(A)      # for symmetric matrices
np.linalg.svd(A)       # most stable decomposition

# Check for near-singularity
if np.linalg.cond(A) > 1e10:
    print("Warning: matrix is ill-conditioned")

# Add regularization for stability
A_reg = A.T @ A + lambda * np.eye(n)
x = np.linalg.solve(A_reg, A.T @ b)
```

### Floating Point Comparison

```python
# ❌ Bad
if det == 0:
    print("singular")

# ✓ Good
if np.abs(det) < 1e-10:
    print("nearly singular")

# Or use numpy's isclose
np.allclose(A, B, rtol=1e-5, atol=1e-8)
```

---

## Common Patterns

### Centering Data

```python
X_centered = X - X.mean(axis=0)
```

### Standardizing (Z-score)

```python
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
```

### Whitening

Transform data to have identity covariance:

```python
def whiten(X):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-8))
    W = eigenvectors @ D_inv_sqrt @ eigenvectors.T
    return X_centered @ W
```

### Low-Rank Approximation

Keep top k singular values:

```python
def low_rank_approx(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Error is sum of discarded singular values squared
# Optimal in Frobenius norm sense
```

### Orthogonalization (Gram-Schmidt)

```python
def gram_schmidt(V):
    """Orthonormalize columns of V."""
    Q = np.zeros_like(V, dtype=float)
    for i in range(V.shape[1]):
        q = V[:, i].astype(float)
        for j in range(i):
            q -= np.dot(Q[:, j], V[:, i]) * Q[:, j]
        Q[:, i] = q / np.linalg.norm(q)
    return Q
```

### Pseudoinverse

For non-square or singular matrices:

```python
A_pinv = np.linalg.pinv(A)

# Properties:
# A @ A_pinv @ A = A
# A_pinv @ A @ A_pinv = A_pinv
# For full-rank A: A_pinv = (A.T @ A)^-1 @ A.T
```

---

## Quick Reference

### NumPy Linear Algebra

```python
# Creation
np.eye(n)                    # identity
np.diag(v)                   # diagonal matrix
np.zeros((m, n))             # zeros
np.random.randn(m, n)        # random normal

# Operations
A @ B                        # matrix multiply
A.T                          # transpose
np.linalg.inv(A)             # inverse
np.linalg.det(A)             # determinant
np.trace(A)                  # trace
np.linalg.norm(v)            # L2 norm
np.linalg.matrix_rank(A)     # rank

# Solving
np.linalg.solve(A, b)        # exact solve Ax=b
np.linalg.lstsq(A, b)        # least squares

# Decompositions
np.linalg.eig(A)             # eigendecomposition
np.linalg.eigh(A)            # for symmetric
np.linalg.svd(A)             # SVD
np.linalg.qr(A)              # QR
np.linalg.cholesky(A)        # Cholesky

# Other
np.linalg.pinv(A)            # pseudoinverse
np.linalg.cond(A)            # condition number
```

### Common Transformations

```python
# 2D rotation (counterclockwise)
R = np.array([[np.cos(θ), -np.sin(θ)],
              [np.sin(θ),  np.cos(θ)]])

# 3D rotation around z-axis
Rz = np.array([[np.cos(θ), -np.sin(θ), 0],
               [np.sin(θ),  np.cos(θ), 0],
               [0,          0,         1]])

# Scaling
S = np.diag([sx, sy, sz])

# Reflection across x-axis
R = np.diag([1, -1])
```
