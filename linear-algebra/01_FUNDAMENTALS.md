# Linear Algebra Fundamentals

The mathematical foundation for machine learning, computer graphics, and robotics.

---

## Table of Contents

1. [Why Linear Algebra Matters](#why-linear-algebra-matters)
2. [Vectors](#vectors)
3. [Matrices](#matrices)
4. [Matrix Operations](#matrix-operations)
5. [Special Matrices](#special-matrices)
6. [Linear Systems](#linear-systems)
7. [Vector Spaces](#vector-spaces)
8. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
9. [Matrix Decompositions](#matrix-decompositions)
10. [Quick Reference](#quick-reference)

---

## Why Linear Algebra Matters

Linear algebra is everywhere in computing:

| Domain | Application |
|--------|-------------|
| **Machine Learning** | Neural networks are just matrix multiplications |
| **Computer Graphics** | Transformations (rotate, scale, translate) |
| **Robotics** | Kinematics, sensor fusion, SLAM |
| **Data Science** | PCA, regression, recommendations |
| **Signal Processing** | Fourier transforms, filtering |
| **Physics Simulations** | Rigid body dynamics, FEM |

---

## Vectors

### What is a Vector?

A vector is an ordered list of numbers representing magnitude and direction.

```
      ┌   ┐
      │ 3 │
v =   │ 4 │   ← 2D vector (column vector)
      └   ┘

      ┌   ┐
      │ 1 │
w =   │ 2 │   ← 3D vector
      │ 3 │
      └   ┘
```

**Notation:**
- **Column vector** (default): n×1 matrix
- **Row vector**: 1×n matrix, written as vᵀ

### Vector Operations

**Addition/Subtraction** (element-wise):
```
┌ 1 ┐   ┌ 4 ┐   ┌ 5 ┐
│ 2 │ + │ 5 │ = │ 7 │
└ 3 ┘   └ 6 ┘   └ 9 ┘
```

**Scalar Multiplication**:
```
    ┌ 1 ┐   ┌ 2 ┐
2 × │ 2 │ = │ 4 │
    └ 3 ┘   └ 6 ┘
```

### Dot Product (Inner Product)

Produces a **scalar**. Measures similarity/projection.

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ

┌ 1 ┐   ┌ 4 ┐
│ 2 │ · │ 5 │ = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
└ 3 ┘   └ 6 ┘
```

**Geometric interpretation:**
```
a · b = |a| × |b| × cos(θ)

where θ is the angle between vectors

• a · b > 0  →  angle < 90° (same general direction)
• a · b = 0  →  angle = 90° (perpendicular/orthogonal)
• a · b < 0  →  angle > 90° (opposite directions)
```

### Cross Product (3D only)

Produces a **vector** perpendicular to both inputs.

```
        ┌ a₂b₃ - a₃b₂ ┐
a × b = │ a₃b₁ - a₁b₃ │
        └ a₁b₂ - a₂b₁ ┘

┌ 1 ┐   ┌ 4 ┐   ┌ 2×6 - 3×5 ┐   ┌ -3 ┐
│ 2 │ × │ 5 │ = │ 3×4 - 1×6 │ = │  6 │
└ 3 ┘   └ 6 ┘   └ 1×5 - 2×4 ┘   └ -3 ┘
```

**Properties:**
- Result is perpendicular to both input vectors
- Magnitude = area of parallelogram formed by the vectors
- Order matters: a × b = -(b × a)

### Vector Norm (Length/Magnitude)

**L2 Norm (Euclidean):**
```
‖v‖₂ = √(v₁² + v₂² + ... + vₙ²)

‖[3, 4]‖₂ = √(9 + 16) = √25 = 5
```

**Other norms:**
```
L1 Norm (Manhattan):  ‖v‖₁ = |v₁| + |v₂| + ... + |vₙ|
L∞ Norm (Max):        ‖v‖∞ = max(|v₁|, |v₂|, ..., |vₙ|)
```

### Unit Vectors

A vector with length 1. Normalize by dividing by the norm:

```
û = v / ‖v‖

v = [3, 4]
‖v‖ = 5
û = [3/5, 4/5] = [0.6, 0.8]
```

### Projection

Project vector a onto vector b:

```
           (a · b)
proj_b(a) = ────── × b
            (b · b)

Or using unit vector:
proj_b(a) = (a · b̂) × b̂
```

```
        b
       ╱
      ╱
     ╱───────→ proj_b(a)
    ╱    ╱
   ╱    ╱ a
  ╱    ╱
 ╱    ↙
```

---

## Matrices

### What is a Matrix?

A 2D array of numbers. An m×n matrix has m rows and n columns.

```
      ┌           ┐
      │ 1   2   3 │
A =   │ 4   5   6 │    ← 2×3 matrix (2 rows, 3 columns)
      └           ┘

Element notation: A[i,j] or aᵢⱼ
A[1,2] = 2  (row 1, column 2, 1-indexed)
```

### Ways to Think About Matrices

**1. As a collection of column vectors:**
```
      ┌     ┐
A =   │ c₁  c₂  c₃ │   where each cᵢ is a column vector
      └     ┘
```

**2. As a collection of row vectors:**
```
      ┌ r₁ ┐
A =   │ r₂ │   where each rᵢ is a row vector
      └ r₃ ┘
```

**3. As a linear transformation:**
```
Av = w

Matrix A transforms vector v into vector w
```

---

## Matrix Operations

### Addition/Subtraction

Element-wise, same dimensions required:

```
┌ 1  2 ┐   ┌ 5  6 ┐   ┌ 6   8 ┐
│ 3  4 │ + │ 7  8 │ = │ 10 12 │
└      ┘   └      ┘   └       ┘
```

### Scalar Multiplication

Multiply every element:

```
    ┌ 1  2 ┐   ┌ 2  4 ┐
2 × │ 3  4 │ = │ 6  8 │
    └      ┘   └      ┘
```

### Matrix Multiplication

**Rule:** (m×n) × (n×p) = (m×p)

Inner dimensions must match. Result has outer dimensions.

```
(2×3) × (3×2) = (2×2)  ✓
(2×3) × (2×3) = error  ✗ (3 ≠ 2)
```

**Computation:** Each element is dot product of row and column.

```
┌ 1  2  3 ┐   ┌ 7   8  ┐   ┌ 1×7+2×9+3×11   1×8+2×10+3×12  ┐
│ 4  5  6 │ × │ 9   10 │ = │ 4×7+5×9+6×11   4×8+5×10+6×12  │
└         ┘   │ 11  12 │   └                                ┘
              └        ┘
            ┌ 58   64  ┐
          = │ 139  154 │
            └          ┘
```

**Properties:**
- NOT commutative: AB ≠ BA (in general)
- Associative: (AB)C = A(BC)
- Distributive: A(B + C) = AB + AC

### Matrix-Vector Multiplication

Special case of matrix multiplication. Transforms a vector.

```
┌ 1  2 ┐   ┌ x ┐   ┌ 1x + 2y ┐
│ 3  4 │ × │ y │ = │ 3x + 4y │
└      ┘   └   ┘   └         ┘
```

**Geometric interpretation:**
```
┌ cos θ  -sin θ ┐   ┌ x ┐
│ sin θ   cos θ │ × │ y │  = rotates (x,y) by angle θ
└               ┘   └   ┘
```

### Transpose

Flip rows and columns:

```
      ┌ 1  2  3 ┐         ┌ 1  4 ┐
A =   │ 4  5  6 │   Aᵀ =  │ 2  5 │
      └         ┘         │ 3  6 │
                          └      ┘

(m×n)ᵀ = (n×m)
```

**Properties:**
- (Aᵀ)ᵀ = A
- (A + B)ᵀ = Aᵀ + Bᵀ
- (AB)ᵀ = BᵀAᵀ  (note the reversal!)
- (cA)ᵀ = cAᵀ

### Element-wise Operations (Hadamard)

Multiply corresponding elements (same dimensions):

```
┌ 1  2 ┐   ┌ 5  6 ┐   ┌ 5   12 ┐
│ 3  4 │ ⊙ │ 7  8 │ = │ 21  32 │
└      ┘   └      ┘   └        ┘
```

---

## Special Matrices

### Identity Matrix (I)

The "1" of matrix multiplication. AI = IA = A.

```
      ┌ 1  0  0 ┐
I₃ =  │ 0  1  0 │
      │ 0  0  1 │
      └         ┘
```

### Diagonal Matrix

Non-zero elements only on the main diagonal:

```
      ┌ 3  0  0 ┐
D =   │ 0  5  0 │     Easy to multiply: just scales each dimension
      │ 0  0  2 │
      └         ┘
```

### Symmetric Matrix

Equals its transpose: A = Aᵀ

```
      ┌ 1  2  3 ┐
A =   │ 2  4  5 │     aᵢⱼ = aⱼᵢ
      │ 3  5  6 │
      └         ┘
```

### Orthogonal Matrix

Columns are orthonormal (perpendicular unit vectors).

```
QᵀQ = QQᵀ = I
Q⁻¹ = Qᵀ     (inverse is just transpose!)

Example (2D rotation):
      ┌ cos θ  -sin θ ┐
Q =   │ sin θ   cos θ │
      └               ┘
```

**Properties:**
- Preserves lengths: ‖Qv‖ = ‖v‖
- Preserves angles
- det(Q) = ±1

### Inverse Matrix

A⁻¹ such that AA⁻¹ = A⁻¹A = I

```
A⁻¹ "undoes" the transformation A

Only exists if:
• Matrix is square (n×n)
• Determinant ≠ 0 (matrix is "invertible" or "non-singular")
```

**2×2 inverse formula:**
```
      ┌ a  b ┐           1      ┌  d  -b ┐
A =   │ c  d │   A⁻¹ = ───── × │ -c   a │
      └      ┘         ad-bc   └        ┘

where (ad - bc) is the determinant
```

### Determinant

A scalar value that encodes "how much" a matrix scales space.

**2×2:**
```
      ┌ a  b ┐
det   │ c  d │ = ad - bc
      └      ┘
```

**3×3 (expansion along first row):**
```
      ┌ a  b  c ┐
det   │ d  e  f │ = a(ei - fh) - b(di - fg) + c(dh - eg)
      │ g  h  i │
      └         ┘
```

**Interpretation:**
- |det(A)| = factor by which A scales area/volume
- det(A) < 0 = A flips orientation
- det(A) = 0 = A collapses dimension (not invertible)

### Rank

The number of linearly independent rows (or columns).

```
Full rank: rank(A) = min(m, n)
           Matrix has maximum possible rank
           Square matrix is invertible iff full rank
```

### Trace

Sum of diagonal elements:

```
tr(A) = a₁₁ + a₂₂ + ... + aₙₙ

Properties:
• tr(A + B) = tr(A) + tr(B)
• tr(AB) = tr(BA)
• tr(A) = sum of eigenvalues
```

---

## Linear Systems

### Solving Ax = b

Find x given matrix A and vector b.

```
┌ 2  1 ┐   ┌ x ┐   ┌ 5 ┐
│ 1  3 │ × │ y │ = │ 5 │
└      ┘   └   ┘   └   ┘

System of equations:
2x + 1y = 5
1x + 3y = 5

Solution: x = 2, y = 1
```

### Solution Cases

```
Ax = b has:

• Unique solution    when A is invertible (full rank)
                     x = A⁻¹b

• No solution        when b is outside column space of A
                     (inconsistent system)

• Infinite solutions when A has dependent columns
                     (underdetermined system)
```

### Gaussian Elimination

Systematic way to solve linear systems:

```
Augmented matrix [A|b]:
┌ 2  1 | 5 ┐
│ 1  3 | 5 │
└          ┘

Row operations to get row echelon form:
┌ 2  1  | 5   ┐
│ 0  2.5| 2.5 │   ← (R2 - 0.5×R1)
└             ┘

Back-substitute:
y = 1
x = (5 - 1) / 2 = 2
```

---

## Vector Spaces

### Span

All possible linear combinations of a set of vectors:

```
span({v₁, v₂}) = {a₁v₁ + a₂v₂ : for all scalars a₁, a₂}

Example:
span({[1,0], [0,1]}) = all of R² (the entire 2D plane)
span({[1,0], [2,0]}) = only the x-axis (vectors are collinear)
```

### Linear Independence

Vectors are linearly independent if none can be written as a combination of others:

```
c₁v₁ + c₂v₂ + ... + cₙvₙ = 0

Only solution is c₁ = c₂ = ... = cₙ = 0

Independent:  {[1,0], [0,1]}     ← can't make one from the other
Dependent:    {[1,0], [2,0], [0,1]} ← [2,0] = 2×[1,0]
```

### Basis

A minimal set of vectors that spans a space:

```
Standard basis for R³:
e₁ = [1, 0, 0]
e₂ = [0, 1, 0]
e₃ = [0, 0, 1]

Any vector in R³ can be written as:
v = a₁e₁ + a₂e₂ + a₃e₃
```

### Column Space and Null Space

**Column Space (Range):**
All possible outputs Ax.
```
Col(A) = span of column vectors of A
```

**Null Space (Kernel):**
All vectors x where Ax = 0.
```
Null(A) = {x : Ax = 0}
```

---

## Eigenvalues and Eigenvectors

### Definition

For a square matrix A, eigenvector v and eigenvalue λ satisfy:

```
Av = λv

The transformation A only SCALES v, doesn't change direction.
```

**Visual:**
```
Before:  →v
After:   ────→ Av = λv  (same direction, different length)
```

### Finding Eigenvalues

Solve the characteristic equation:

```
det(A - λI) = 0

For 2×2:
A = ┌ a  b ┐
    │ c  d │

det(A - λI) = (a-λ)(d-λ) - bc = 0
λ² - (a+d)λ + (ad-bc) = 0
λ² - tr(A)λ + det(A) = 0
```

### Finding Eigenvectors

For each eigenvalue λ, solve:

```
(A - λI)v = 0

Find the null space of (A - λI)
```

### Example

```
A = ┌ 4  1 ┐
    │ 2  3 │

Characteristic equation:
det(A - λI) = (4-λ)(3-λ) - 2 = λ² - 7λ + 10 = (λ-5)(λ-2) = 0

Eigenvalues: λ₁ = 5, λ₂ = 2

For λ₁ = 5:
(A - 5I)v = 0
┌ -1  1 ┐   ┌ x ┐   ┌ 0 ┐
│  2 -2 │ × │ y │ = │ 0 │
-x + y = 0  →  v₁ = [1, 1]

For λ₂ = 2:
(A - 2I)v = 0
┌ 2  1 ┐   ┌ x ┐   ┌ 0 ┐
│ 2  1 │ × │ y │ = │ 0 │
2x + y = 0  →  v₂ = [1, -2]
```

### Properties

```
• Sum of eigenvalues = trace(A)
• Product of eigenvalues = det(A)
• A matrix is invertible iff all eigenvalues ≠ 0
• Symmetric matrices have real eigenvalues and orthogonal eigenvectors
```

### Applications

| Application | How Eigenvalues Help |
|-------------|---------------------|
| **PCA** | Eigenvectors of covariance matrix = principal components |
| **PageRank** | Dominant eigenvector of link matrix |
| **Vibrations** | Eigenvalues = natural frequencies |
| **Stability** | System stable if all eigenvalues have negative real part |
| **Markov Chains** | Steady state = eigenvector for λ=1 |

---

## Matrix Decompositions

### Eigendecomposition

For matrices with n linearly independent eigenvectors:

```
A = VΛV⁻¹

Where:
V = matrix of eigenvectors (as columns)
Λ = diagonal matrix of eigenvalues
```

**Power of decomposition:**
```
A² = VΛV⁻¹ × VΛV⁻¹ = VΛ²V⁻¹
Aⁿ = VΛⁿV⁻¹   ← Λⁿ is trivial (just raise each diagonal element)
```

### Singular Value Decomposition (SVD)

Works for ANY matrix (not just square):

```
A = UΣVᵀ

Where:
• A is m×n
• U is m×m orthogonal (left singular vectors)
• Σ is m×n diagonal (singular values, non-negative)
• V is n×n orthogonal (right singular vectors)
```

**Interpretation:**
```
Any linear transformation = rotation (V) → scale (Σ) → rotation (U)
```

**Applications:**
- Dimensionality reduction (keep top k singular values)
- Image compression
- Recommender systems
- Pseudoinverse computation

### QR Decomposition

Decompose into orthogonal × upper triangular:

```
A = QR

Where:
• Q is orthogonal (Qᵀ = Q⁻¹)
• R is upper triangular
```

**Uses:**
- Solving least squares
- Eigenvalue algorithms
- Gram-Schmidt orthogonalization

### Cholesky Decomposition

For symmetric positive definite matrices:

```
A = LLᵀ

Where L is lower triangular.
```

**Uses:**
- Efficient solving of Ax = b
- Sampling from multivariate Gaussians
- Numerical stability

---

## Quick Reference

### Vector Operations

| Operation | Formula | Result |
|-----------|---------|--------|
| Dot product | a·b = Σaᵢbᵢ | Scalar |
| Cross product | a×b | Vector (3D) |
| Norm | ‖v‖ = √(Σvᵢ²) | Scalar |
| Normalize | v/‖v‖ | Unit vector |
| Angle | cos θ = (a·b)/(‖a‖‖b‖) | Radians |

### Matrix Properties

| Property | Notation | Definition |
|----------|----------|------------|
| Transpose | Aᵀ | Flip rows/columns |
| Inverse | A⁻¹ | AA⁻¹ = I |
| Determinant | det(A) | Scaling factor |
| Trace | tr(A) | Sum of diagonal |
| Rank | rank(A) | # independent rows/cols |

### Key Formulas

```
(AB)ᵀ = BᵀAᵀ
(AB)⁻¹ = B⁻¹A⁻¹
det(AB) = det(A)det(B)
det(A⁻¹) = 1/det(A)
det(Aᵀ) = det(A)

For 2×2 inverse:
[a b]⁻¹ =  1/(ad-bc) × [ d -b]
[c d]                   [-c  a]
```

### Eigenvalue Summary

```
Av = λv                  Definition
det(A - λI) = 0          Find eigenvalues
(A - λI)v = 0            Find eigenvectors
Σλᵢ = tr(A)              Sum = trace
Πλᵢ = det(A)             Product = determinant
```

---

Next: [Practical Linear Algebra](./02_PRACTICAL.md) - Applications in ML, graphics, and robotics
