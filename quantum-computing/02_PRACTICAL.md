# Practical Quantum Computing

Programming quantum computers, simulators, and real-world applications.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Qiskit (IBM)](#qiskit-ibm)
3. [Cirq (Google)](#cirq-google)
4. [Other Frameworks](#other-frameworks)
5. [Running on Real Hardware](#running-on-real-hardware)
6. [Common Patterns & Recipes](#common-patterns--recipes)
7. [NISQ Algorithms](#nisq-algorithms)
8. [Quantum Machine Learning](#quantum-machine-learning)
9. [Post-Quantum Cryptography](#post-quantum-cryptography)
10. [Resources & Next Steps](#resources--next-steps)

---

## Getting Started

### Which Framework?

| Framework | Provider | Best For |
|-----------|----------|----------|
| **Qiskit** | IBM | Beginners, large community, real hardware |
| **Cirq** | Google | Research, custom circuits |
| **PennyLane** | Xanadu | Quantum ML, differentiable |
| **Amazon Braket** | AWS | Multi-vendor access |
| **Q#** | Microsoft | Quantum algorithms research |

### Installation

```bash
# Qiskit (recommended for beginners)
pip install qiskit qiskit-aer qiskit-ibm-runtime

# Cirq
pip install cirq

# PennyLane
pip install pennylane

# All of them work with Python 3.9+
```

---

## Qiskit (IBM)

### Your First Quantum Circuit

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Create a 2-qubit circuit
qc = QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits

# Build the circuit
qc.h(0)         # Hadamard on qubit 0: creates superposition
qc.cx(0, 1)     # CNOT: entangles qubits 0 and 1
qc.measure([0, 1], [0, 1])  # Measure both qubits

# Visualize
print(qc.draw())
```

Output:
```
     ┌───┐     ┌─┐
q_0: ┤ H ├──●──┤M├───
     └───┘┌─┴─┐└╥┘┌─┐
q_1: ─────┤ X ├─╫─┤M├
          └───┘ ║ └╥┘
c: 2/═══════════╩══╩═
                0  1
```

### Running a Simulation

```python
from qiskit_aer import AerSimulator

# Create simulator
simulator = AerSimulator()

# Run circuit 1000 times
job = simulator.run(qc, shots=1000)
result = job.result()
counts = result.get_counts()

print(counts)
# Output: {'00': 502, '11': 498}
# Bell state: always measure 00 or 11 (never 01 or 10)
```

### Common Gates in Qiskit

```python
qc = QuantumCircuit(2)

# Single-qubit gates
qc.h(0)           # Hadamard
qc.x(0)           # Pauli-X (NOT)
qc.y(0)           # Pauli-Y
qc.z(0)           # Pauli-Z
qc.s(0)           # S gate (√Z)
qc.t(0)           # T gate (√S)
qc.rx(3.14, 0)    # Rotation around X by π
qc.ry(1.57, 0)    # Rotation around Y by π/2
qc.rz(0.5, 0)     # Rotation around Z

# Two-qubit gates
qc.cx(0, 1)       # CNOT (control=0, target=1)
qc.cz(0, 1)       # Controlled-Z
qc.swap(0, 1)     # Swap qubits
qc.cp(3.14, 0, 1) # Controlled phase

# Three-qubit gates
qc.ccx(0, 1, 2)   # Toffoli (CCX)
qc.cswap(0, 1, 2) # Fredkin (controlled-SWAP)
```

### Statevector Analysis

```python
from qiskit.quantum_info import Statevector

# Create circuit without measurement
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Get exact statevector
statevector = Statevector(qc)
print(statevector)
# Statevector([0.707+0.j, 0.+0.j, 0.+0.j, 0.707+0.j])
# This is (|00⟩ + |11⟩)/√2

# Visualize
statevector.draw('latex')  # or 'text', 'bloch'
```

### Building Parameterized Circuits

```python
from qiskit.circuit import Parameter
import numpy as np

# Create parameter
theta = Parameter('θ')

# Build parameterized circuit
qc = QuantumCircuit(1, 1)
qc.ry(theta, 0)
qc.measure(0, 0)

# Bind parameter to specific value
bound_circuit = qc.assign_parameters({theta: np.pi/4})
```

---

## Cirq (Google)

### Basic Circuit

```python
import cirq

# Create qubits
q0, q1 = cirq.LineQubit.range(2)

# Build circuit
circuit = cirq.Circuit([
    cirq.H(q0),           # Hadamard
    cirq.CNOT(q0, q1),    # Entangle
    cirq.measure(q0, q1, key='result')
])

print(circuit)
```

Output:
```
0: ───H───@───M('result')───
          │   │
1: ───────X───M─────────────
```

### Simulation

```python
# Simulate
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)

# Get histogram
print(result.histogram(key='result'))
# Counter({0: 498, 3: 502})
# 0 = 00 binary, 3 = 11 binary
```

### Cirq vs Qiskit

| Aspect | Qiskit | Cirq |
|--------|--------|------|
| Qubit naming | Integer indices | Named objects |
| Circuit style | Builder pattern | List of moments |
| Gate naming | `cx`, `h`, `t` | `CNOT`, `H`, `T` |
| Measurement | Separate classical bits | Key-based |
| Visualization | ASCII & matplotlib | ASCII & SVG |

---

## Other Frameworks

### PennyLane (Quantum ML)

```python
import pennylane as qml
from pennylane import numpy as np

# Create device
dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[1], wires=1)
    return qml.expval(qml.PauliZ(0))

# This is differentiable!
params = np.array([0.5, 0.1], requires_grad=True)
gradient = qml.grad(circuit)(params)
print(f"Gradient: {gradient}")
```

### Amazon Braket

```python
from braket.circuits import Circuit
from braket.aws import AwsDevice

# Build circuit
circuit = Circuit().h(0).cnot(0, 1)

# Run on simulator
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
task = device.run(circuit, shots=1000)
result = task.result()
print(result.measurement_counts)
```

---

## Running on Real Hardware

### IBM Quantum (Free Tier)

```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# Save credentials (once)
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")

# Connect
service = QiskitRuntimeService()

# Select backend
backend = service.least_busy(operational=True, simulator=False)
print(f"Running on: {backend.name}")

# Create circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Transpile for specific hardware
from qiskit import transpile
transpiled = transpile(qc, backend)

# Run
sampler = SamplerV2(backend)
job = sampler.run([transpiled], shots=1000)
result = job.result()
```

### Understanding Hardware Constraints

```
Real quantum hardware has:

1. LIMITED CONNECTIVITY
   Not all qubits can directly interact
   
   Ideal:          Reality (heavy-hex):
   ●─●─●─●           ●───●───●
   │ │ │ │           │   │   │
   ●─●─●─●           ●───●───●
   │ │ │ │               │
   ●─●─●─●           ●───●───●

2. NOISY GATES
   Every gate has error probability
   1-qubit: ~0.01-0.1% error
   2-qubit: ~0.5-2% error
   
3. LIMITED COHERENCE TIME
   Qubits lose quantum state
   T1 (relaxation): ~100 μs
   T2 (dephasing): ~50-100 μs
   
4. MEASUREMENT ERRORS
   Reading wrong value: ~1-5%
```

### Transpilation: Adapting to Hardware

```python
from qiskit import transpile

# Original circuit (ideal)
qc = QuantumCircuit(3)
qc.ccx(0, 1, 2)  # Toffoli gate

# Transpile for real hardware
transpiled = transpile(qc, backend, optimization_level=3)

print(f"Original gates: {qc.depth()}")
print(f"Transpiled gates: {transpiled.depth()}")
# Toffoli → many 1-qubit and CNOT gates
```

---

## Common Patterns & Recipes

### Create Uniform Superposition

```python
def uniform_superposition(n_qubits):
    """Create |+⟩^n = equal superposition of all basis states."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    return qc

# 3 qubits → equal superposition of |000⟩ through |111⟩
```

### Create Bell Pairs

```python
def bell_pair(qc, q0, q1):
    """Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
    qc.h(q0)
    qc.cx(q0, q1)
    return qc
```

### Create GHZ State

```python
def ghz_state(n_qubits):
    """Create (|00...0⟩ + |11...1⟩)/√2 across n qubits."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc
```

### Quantum Fourier Transform

```python
import numpy as np

def qft(qc, n):
    """Apply QFT to first n qubits."""
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(np.pi / 2**(j - i), j, i)
    # Swap qubits to match conventional ordering
    for i in range(n // 2):
        qc.swap(i, n - i - 1)
    return qc
```

### Grover's Oracle (for specific target)

```python
def grover_oracle(qc, target_state):
    """Mark target state with phase flip."""
    n = qc.num_qubits
    
    # Flip qubits where target has 0
    for i, bit in enumerate(reversed(target_state)):
        if bit == '0':
            qc.x(i)
    
    # Multi-controlled Z
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)  # Toffoli chain
    qc.h(n - 1)
    
    # Undo flips
    for i, bit in enumerate(reversed(target_state)):
        if bit == '0':
            qc.x(i)
    
    return qc
```

### Grover Diffusion Operator

```python
def diffusion(qc, n):
    """Grover diffusion (reflection about mean)."""
    for i in range(n):
        qc.h(i)
        qc.x(i)
    
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    
    for i in range(n):
        qc.x(i)
        qc.h(i)
    
    return qc
```

---

## NISQ Algorithms

### Variational Quantum Eigensolver (VQE)

```python
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

# Define Hamiltonian (e.g., H2 molecule simplified)
hamiltonian = SparsePauliOp.from_list([
    ("II", -1.052373245772859),
    ("IZ", 0.39793742484318045),
    ("ZI", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX", 0.18093119978423156)
])

# Ansatz (parameterized circuit)
ansatz = TwoLocal(2, ['ry', 'rz'], 'cz', reps=2)

# Run VQE
estimator = Estimator()
optimizer = COBYLA(maxiter=500)
vqe = VQE(estimator, ansatz, optimizer)
result = vqe.compute_minimum_eigenvalue(hamiltonian)

print(f"Ground state energy: {result.eigenvalue:.6f}")
```

### Quantum Approximate Optimization (QAOA)

```python
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

# Define cost Hamiltonian for MaxCut problem
# Example: 3-node graph
cost_op = SparsePauliOp.from_list([
    ("ZZI", 0.5),
    ("ZIZ", 0.5),
    ("IZZ", 0.5)
])

# Run QAOA
sampler = Sampler()
qaoa = QAOA(sampler, optimizer=COBYLA(maxiter=100), reps=2)
result = qaoa.compute_minimum_eigenvalue(cost_op)

print(f"Best solution: {result.best_measurement}")
```

---

## Quantum Machine Learning

### Quantum Kernel Methods

```python
import pennylane as qml
from pennylane import numpy as np

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def quantum_kernel(x1, x2):
    """Compute kernel between two data points."""
    # Encode first data point
    qml.RY(x1[0], wires=0)
    qml.RY(x1[1], wires=1)
    
    # Encode second data point (adjoint)
    qml.adjoint(qml.RY)(x2[1], wires=1)
    qml.adjoint(qml.RY)(x2[0], wires=0)
    
    return qml.probs(wires=[0, 1])

# Kernel value = probability of measuring |00⟩
x1, x2 = np.array([0.5, 0.3]), np.array([0.4, 0.6])
k = quantum_kernel(x1, x2)[0]  # Get |00⟩ probability
print(f"Kernel value: {k}")
```

### Variational Quantum Classifier

```python
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def quantum_classifier(weights, x):
    # Data encoding
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    
    # Variational layers
    for layer_weights in weights:
        qml.RY(layer_weights[0], wires=0)
        qml.RY(layer_weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
    
    return qml.expval(qml.PauliZ(0))

def cost(weights, X, Y):
    predictions = [quantum_classifier(weights, x) for x in X]
    return np.mean((np.array(predictions) - Y) ** 2)

# Training
weights = np.random.randn(3, 2, requires_grad=True)
opt = AdamOptimizer(0.1)

for epoch in range(100):
    weights = opt.step(lambda w: cost(w, X_train, y_train), weights)
```

### Quantum Neural Network Architectures

```
Common QML Ansätze:

1. HARDWARE-EFFICIENT ANSATZ
   ┌────┐┌────┐     ┌────┐┌────┐
   ┤ Ry ├┤ Rz ├──●──┤ Ry ├┤ Rz ├──●──
   └────┘└────┘  │  └────┘└────┘  │
   ┌────┐┌────┐┌─┴─┐┌────┐┌────┐┌─┴─┐
   ┤ Ry ├┤ Rz ├┤ X ├┤ Ry ├┤ Rz ├┤ X ├
   └────┘└────┘└───┘└────┘└────┘└───┘
   
2. STRONGLY ENTANGLING
   More connectivity, deeper entanglement
   
3. TREE TENSOR NETWORK
   Hierarchical structure, good for correlations
```

---

## Post-Quantum Cryptography

### The Threat

```
Current encryption:          Quantum threat:

RSA-2048                     Shor's algorithm
• Factor n = p × q           • O((log n)³) on quantum
• Takes classical ~10^14 yr  • Could break in hours
                             with ~4000 logical qubits

Timeline estimate:
2030-2035: "Q-Day" - RSA/ECC broken by quantum
```

### NIST Post-Quantum Standards (2024)

| Algorithm | Type | Use Case |
|-----------|------|----------|
| **ML-KEM (Kyber)** | Lattice | Key encapsulation |
| **ML-DSA (Dilithium)** | Lattice | Digital signatures |
| **SLH-DSA (SPHINCS+)** | Hash-based | Signatures (conservative) |
| **FN-DSA (FALCON)** | Lattice | Compact signatures |

### Using Post-Quantum Crypto (Python)

```python
# Using liboqs (Open Quantum Safe)
# pip install liboqs-python

import oqs

# Key encapsulation (Kyber)
kem = oqs.KeyEncapsulation("Kyber512")
public_key = kem.generate_keypair()
ciphertext, shared_secret = kem.encap_secret(public_key)
decrypted_secret = kem.decap_secret(ciphertext)

# Digital signature (Dilithium)
sig = oqs.Signature("Dilithium2")
public_key = sig.generate_keypair()
signature = sig.sign(b"Message to sign")
is_valid = sig.verify(b"Message to sign", signature, public_key)
```

### Hybrid Approach (Transition Period)

```
Current best practice: Combine classical + post-quantum

TLS Handshake:
┌─────────────────────────────────────────────────────┐
│ ECDH (classical) + Kyber (post-quantum)             │
│                                                     │
│ Shared Secret = ECDH_secret || Kyber_secret         │
│                                                     │
│ Benefits:                                           │
│ • Secure if EITHER algorithm is unbroken            │
│ • Backward compatible                               │
│ • Already deployed in Chrome, Cloudflare            │
└─────────────────────────────────────────────────────┘
```

---

## Resources & Next Steps

### Learning Path

```
Beginner:
1. IBM Quantum Learning: learning.quantum-computing.ibm.com
2. Qiskit Textbook: qiskit.org/textbook
3. Quantum Country (Andy Matuschak): quantum.country

Intermediate:
4. "Quantum Computing: An Applied Approach" - Hidary
5. Cirq tutorials: quantumai.google/cirq
6. PennyLane demos: pennylane.ai/qml

Advanced:
7. "Quantum Computation and Quantum Information" - Nielsen & Chuang
8. arXiv.org quant-ph (research papers)
9. Contribute to open source (Qiskit, Cirq, PennyLane)
```

### Practice Platforms

| Platform | Best For |
|----------|----------|
| **IBM Quantum** | Real hardware, learning |
| **Quirk** | Visualizing circuits (browser) |
| **Amazon Braket** | Multi-vendor |
| **Azure Quantum** | Optimization, Q# |
| **Quantum Katas** | Q# exercises |

### Online Simulators

```
Free browser-based:

1. IBM Quantum Composer
   quantum-computing.ibm.com/composer
   • Drag-and-drop circuit building
   • Real hardware access

2. Quirk
   algassert.com/quirk
   • Real-time statevector display
   • Great for intuition

3. Quantum Playground
   quantumplayground.net
   • GPU-accelerated simulation
```

### Keeping Up

```
Stay current:

• @IBM_Quantum, @GoogleAI on Twitter/X
• r/QuantumComputing on Reddit
• Qiskit Slack community
• Unitary Fund newsletter
• The Quantum Insider (news)
```

---

## Quick Reference

### Qiskit Cheat Sheet

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# Create circuit
qc = QuantumCircuit(n_qubits, n_classical_bits)

# Gates
qc.h(qubit)           # Hadamard
qc.x(qubit)           # NOT
qc.cx(ctrl, tgt)      # CNOT
qc.measure(q, c)      # Measure qubit q → classical bit c
qc.measure_all()      # Measure all qubits

# Simulate
sim = AerSimulator()
result = sim.run(qc, shots=1000).result()
counts = result.get_counts()

# Statevector
sv = Statevector(qc)
probs = sv.probabilities()

# Visualize
qc.draw('mpl')        # Matplotlib
qc.draw('text')       # ASCII
```

### Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `CircuitError: not enough classical bits` | Missing measurement targets | Add classical bits: `QuantumCircuit(2, 2)` |
| `TranspilerError` | Circuit incompatible with backend | Use `transpile(qc, backend)` |
| `IBMQAccountError` | Token issue | Re-save account credentials |
| Gate on non-existent qubit | Index out of range | Check `qc.num_qubits` |

### Performance Tips

```python
# 1. Minimize circuit depth
# Shallow circuits = less decoherence

# 2. Use native gates
# Transpiler converts gates; start with hardware-native

# 3. Optimize measurement
# Measure only what you need

# 4. Use efficient simulation
from qiskit_aer import AerSimulator
sim = AerSimulator(method='statevector')  # For small circuits
sim = AerSimulator(method='matrix_product_state')  # For sparse

# 5. Batch jobs
# Submit multiple circuits in one job
results = sim.run([qc1, qc2, qc3]).result()
```

---

## Key Takeaways

1. **Start with Qiskit** — best documentation and community
2. **Simulate first** — real hardware is noisy and limited
3. **Understand transpilation** — circuits must match hardware
4. **NISQ algorithms** — VQE, QAOA work on today's hardware
5. **Quantum ML is hybrid** — classical optimizer + quantum circuit
6. **Post-quantum crypto is now** — migration should start today
7. **Noise is the enemy** — error mitigation is essential
8. **Real applications are emerging** — chemistry, optimization, ML

---

## Example: Complete Grover's Search

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

def grovers_algorithm(target: str):
    """Search for target bit string."""
    n = len(target)
    qc = QuantumCircuit(n, n)
    
    # Initialize superposition
    qc.h(range(n))
    
    # Number of iterations
    iterations = int(np.pi / 4 * np.sqrt(2**n))
    
    for _ in range(iterations):
        # Oracle: mark target
        for i, bit in enumerate(reversed(target)):
            if bit == '0':
                qc.x(i)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for i, bit in enumerate(reversed(target)):
            if bit == '0':
                qc.x(i)
        
        # Diffusion
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))
    
    qc.measure(range(n), range(n))
    
    # Run
    sim = AerSimulator()
    result = sim.run(qc, shots=1000).result()
    counts = result.get_counts()
    
    # Find most common result
    found = max(counts, key=counts.get)
    return found, counts

# Search for "101"
target = "101"
found, counts = grovers_algorithm(target)
print(f"Searching for: {target}")
print(f"Found: {found}")
print(f"Counts: {counts}")
# Should find "101" with high probability (~900+ out of 1000)
```
