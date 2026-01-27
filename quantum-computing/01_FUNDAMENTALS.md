# Quantum Computing Fundamentals

From classical bits to quantum supremacy—the essential concepts without drowning in physics.

---

## Table of Contents

1. [Quick Physics Primer](#quick-physics-primer)
2. [Classical vs Quantum Computing](#classical-vs-quantum-computing)
3. [Qubits](#qubits)
4. [Superposition](#superposition)
5. [Measurement](#measurement)
6. [Entanglement](#entanglement)
7. [Quantum Gates](#quantum-gates)
8. [Quantum Circuits](#quantum-circuits)
9. [Key Quantum Algorithms](#key-quantum-algorithms)
10. [Error Correction & Noise](#error-correction--noise)
11. [Current State of Hardware](#current-state-of-hardware)

---

## Quick Physics Primer

Just enough quantum physics to understand quantum computing. Skip if you're eager to get to the computing part.

### Wave-Particle Duality

Light and matter behave as both waves AND particles:

```
Classical thinking:          Quantum reality:
                             
  Particle: •                  Wave: ∿∿∿∿∿
  Localized, definite             Spread out, probabilistic
                             
  OR                           BOTH at once!
                             
  Wave: ∿∿∿∿∿                 Electron going through slits:
  Spread out                   
                               ┌─────┐
                             ──┤     ├──●  Detected as particle
                               │  █  │
                             ──┤     ├──   But interference pattern
                               └─────┘     shows wave behavior!
```

### The Double-Slit Experiment

The most famous quantum experiment:

```
Single slit:                Double slit:
                            
   │█│                         │█│ │█│
   │ │                         │ │ │ │
───┼─┼──▶                   ───┼─┼─┼─┼──▶
   │ │     Simple blob         │ │ │ │     Interference pattern!
   │█│                         │█│ │█│
                            
                               ████████████
Screen:  ████                  █  █  █  █
         ████                  ████████████

Even single particles create interference—
they interfere with themselves!
```

### Probability Amplitudes

Quantum states have **amplitudes** (complex numbers), not just probabilities:

```
Classical probability:    Quantum amplitude:
                          
P(heads) = 0.5            α = complex number
P(tails) = 0.5            
                          Probability = |α|²
P(heads) + P(tails) = 1   
                          Amplitudes can:
                          • Be negative
                          • Be complex (imaginary)
                          • Cancel out (interference)
                          • Add up (constructive)
```

**Why this matters**: Quantum algorithms work by making wrong answers interfere destructively and right answers interfere constructively.

### Heisenberg Uncertainty Principle

You can't know everything precisely at once:

```
Position × Momentum ≥ ℏ/2

The more precisely you know position,
the less precisely you can know momentum.

This isn't about measurement limitations—
it's fundamental to nature.
```

---

## Classical vs Quantum Computing

### The Core Difference

```
┌─────────────────────────────────────────────────────────────────┐
│ Classical Computer                                               │
│                                                                  │
│   Bit: 0 OR 1                                                   │
│                                                                  │
│   ┌─────┐                                                       │
│   │  0  │  Definitely 0, or definitely 1                        │
│   └─────┘  One state at a time                                  │
│                                                                  │
│   n bits = ONE of 2ⁿ states                                     │
│   3 bits = 000, 001, 010... (one at a time)                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ Quantum Computer                                                 │
│                                                                  │
│   Qubit: 0 AND 1 (superposition!)                               │
│                                                                  │
│   ┌─────┐                                                       │
│   │ 0+1 │  Both states simultaneously                           │
│   └─────┘  Until measured                                       │
│                                                                  │
│   n qubits = ALL 2ⁿ states at once                              │
│   3 qubits = 000, 001, 010... (all simultaneously!)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### What Quantum Computers Are Good At

| Good For | Why | Examples |
|----------|-----|----------|
| **Factoring large numbers** | Shor's algorithm | Breaking RSA encryption |
| **Searching unsorted data** | Grover's algorithm | Database search |
| **Optimization problems** | Quantum annealing | Logistics, scheduling |
| **Simulating quantum systems** | Natural fit | Drug discovery, materials |
| **Machine learning** | Quantum speedups | Pattern recognition |

### What Quantum Computers Are NOT Good At

| Not Great For | Why |
|---------------|-----|
| Running Excel | No speedup over classical |
| Web browsing | No quantum advantage |
| Video games | Classical is fine |
| Most everyday tasks | Overhead not worth it |

**Key insight**: Quantum computers are **co-processors** for specific problems, not replacements for classical computers.

---

## Qubits

### Physical Implementations

```
┌─────────────────────────────────────────────────────────────────┐
│ Different ways to make a qubit:                                  │
│                                                                  │
│  Superconducting circuits (IBM, Google)                         │
│  ┌──────────────────────────┐                                   │
│  │ Josephson junction       │  • Fastest gates (~10ns)          │
│  │ cooled to 15 millikelvin │  • Need extreme cooling           │
│  └──────────────────────────┘  • Current leader in qubit count  │
│                                                                  │
│  Trapped ions (IonQ, Honeywell/Quantinuum)                      │
│  ┌──────────────────────────┐                                   │
│  │ Individual atoms held    │  • Highest fidelity               │
│  │ by electromagnetic fields│  • All-to-all connectivity        │
│  └──────────────────────────┘  • Slower gates                   │
│                                                                  │
│  Photonic (Xanadu, PsiQuantum)                                  │
│  ┌──────────────────────────┐                                   │
│  │ Single photons of light  │  • Room temperature!              │
│  │ in optical circuits      │  • Natural for networking         │
│  └──────────────────────────┘  • Hard to make interact          │
│                                                                  │
│  Neutral atoms (QuEra)                                          │
│  ┌──────────────────────────┐                                   │
│  │ Atoms held by laser      │  • Scalable                       │
│  │ tweezers                 │  • Flexible connectivity          │
│  └──────────────────────────┘  • Emerging technology            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The Bloch Sphere

A qubit's state visualized as a point on a sphere:

```
                    |0⟩
                     ●
                    /|\
                   / | \
                  /  |  \
                 /   |   \
                /    |    \
        |+⟩ ●──/─────●─────\──● |-⟩
              \      |      /
               \     |     /
                \    |    /
                 \   |   /
                  \  |  /
                   \ | /
                    \|/
                     ●
                    |1⟩

North pole: |0⟩
South pole: |1⟩
Equator: Equal superpositions with different phases
Any point: A valid qubit state

|+⟩ = (|0⟩ + |1⟩) / √2
|-⟩ = (|0⟩ - |1⟩) / √2
```

---

## Superposition

### The Concept

A qubit can be in a **combination** of 0 and 1:

```
|ψ⟩ = α|0⟩ + β|1⟩

Where:
  α, β are complex numbers (amplitudes)
  |α|² + |β|² = 1 (probabilities sum to 1)
  |α|² = probability of measuring 0
  |β|² = probability of measuring 1
```

### Visualizing Superposition

```
Classical bit:           Qubit in superposition:

     0     1                  0 ──●── 1
     ●     ○             
                              In BOTH states
  Definitely               with amplitudes α and β
   one or the
     other

Equal superposition: |+⟩ = (|0⟩ + |1⟩) / √2
                          
When measured:  50% chance → 0
                50% chance → 1
```

### The Power: Exponential States

```
n qubits can represent 2ⁿ states simultaneously:

1 qubit:  2 states    (α₀|0⟩ + α₁|1⟩)
2 qubits: 4 states    (α₀|00⟩ + α₁|01⟩ + α₂|10⟩ + α₃|11⟩)
3 qubits: 8 states    
...
50 qubits: 2⁵⁰ ≈ 10¹⁵ states!

That's more states than a classical computer can store!
```

**But**: You only get ONE answer when you measure. The art of quantum algorithms is setting up interference so the right answer is likely.

---

## Measurement

### The Collapse

```
Before measurement:           After measurement:

|ψ⟩ = α|0⟩ + β|1⟩              Either |0⟩ or |1⟩
                               
     0 ──●── 1                      ●     OR     ●
                                    0            1
  Superposition                 Definite state
  (both states)                (probability |α|² or |β|²)

Measurement is IRREVERSIBLE—
the superposition is destroyed!
```

### Measurement Bases

You can measure in different "bases":

```
Computational basis (Z):        Hadamard basis (X):

|0⟩ and |1⟩                     |+⟩ and |-⟩

        |0⟩                            |+⟩
         ●                              ●
         |                              |
         |                     ─────────●─────────
         |                              |
         ●                              ●
        |1⟩                            |-⟩

Measuring |+⟩ in Z-basis:     Measuring |0⟩ in X-basis:
  50% → |0⟩                     50% → |+⟩
  50% → |1⟩                     50% → |-⟩
```

---

## Entanglement

### "Spooky Action at a Distance"

Two qubits become correlated in a way impossible classically:

```
Bell State (maximally entangled):

|Φ+⟩ = (|00⟩ + |11⟩) / √2

     Qubit A          Qubit B
        ●─────────────────●
              linked!

If you measure A and get 0 → B MUST be 0
If you measure A and get 1 → B MUST be 1

Even if they're light-years apart!
Correlation is instantaneous (but can't send information faster than light)
```

### Creating Entanglement

```
Start: |00⟩ (both qubits in |0⟩)

Step 1: Apply H (Hadamard) to first qubit
        |00⟩ → (|0⟩ + |1⟩)|0⟩/√2 = (|00⟩ + |10⟩)/√2

Step 2: Apply CNOT (controlled-NOT)
        (|00⟩ + |10⟩)/√2 → (|00⟩ + |11⟩)/√2 = |Φ+⟩

      ┌───┐
q0: ──┤ H ├──●──
      └───┘  │
           ┌─┴─┐
q1: ───────┤ X ├──
           └───┘
```

### Why Entanglement Matters

| Use Case | How Entanglement Helps |
|----------|----------------------|
| **Quantum teleportation** | Transfer qubit state using entanglement + 2 classical bits |
| **Superdense coding** | Send 2 classical bits using 1 entangled qubit |
| **Quantum key distribution** | Detect eavesdroppers (measurement disturbs entanglement) |
| **Quantum algorithms** | Enable correlations impossible classically |

---

## Quantum Gates

### Single-Qubit Gates

| Gate | Matrix | Effect | Circuit |
|------|--------|--------|---------|
| **X (NOT)** | `[0 1; 1 0]` | Flip: \|0⟩↔\|1⟩ | `──[X]──` |
| **Z** | `[1 0; 0 -1]` | Phase flip: \|1⟩→-\|1⟩ | `──[Z]──` |
| **H (Hadamard)** | `[1 1; 1 -1]/√2` | Create superposition | `──[H]──` |
| **S** | `[1 0; 0 i]` | π/2 phase | `──[S]──` |
| **T** | `[1 0; 0 e^(iπ/4)]` | π/4 phase | `──[T]──` |

### Hadamard Gate (The Most Important)

```
       ┌───┐
|0⟩ ───┤ H ├─── |+⟩ = (|0⟩ + |1⟩)/√2
       └───┘

       ┌───┐
|1⟩ ───┤ H ├─── |-⟩ = (|0⟩ - |1⟩)/√2
       └───┘

Creates equal superposition from basis states.
Apply twice: H·H = I (identity, back to original)
```

### Two-Qubit Gates

**CNOT (Controlled-NOT)**: Flip target if control is |1⟩

```
Control ──●──     |00⟩ → |00⟩
          │       |01⟩ → |01⟩
Target  ──⊕──     |10⟩ → |11⟩  (flipped!)
                  |11⟩ → |10⟩  (flipped!)
```

**CZ (Controlled-Z)**: Apply Z to target if control is |1⟩

```
Control ──●──     |00⟩ → |00⟩
          │       |01⟩ → |01⟩
Target  ──●──     |10⟩ → |10⟩
                  |11⟩ → -|11⟩  (phase flip!)
```

**SWAP**: Exchange two qubits

```
q0 ──╳──        |01⟩ → |10⟩
     │
q1 ──╳──
```

### Universal Gate Sets

Any quantum computation can be built from:

```
Option 1: {H, T, CNOT}
Option 2: {H, Toffoli}
Option 3: {Rx, Ry, CNOT} (common in hardware)

These are "universal" - like NAND gates for classical computing.
```

---

## Quantum Circuits

### Reading Circuits

```
     ┌───┐     ┌───┐
q0: ─┤ H ├──●──┤ X ├───
     └───┘  │  └───┘
            │
     ┌───┐┌─┴─┐
q1: ─┤ X ├┤ X ├────────
     └───┘└───┘

Reading left to right:
1. Apply X to q1, H to q0 (parallel operations)
2. Apply CNOT with q0 as control, q1 as target
3. Apply X to q0

Time flows left → right
Each horizontal line is a qubit
```

### Common Patterns

**Quantum parallelism (apply H to all):**

```
     ┌───┐
q0: ─┤ H ├─    From |000⟩ to equal superposition
     └───┘     of all 8 basis states
     ┌───┐
q1: ─┤ H ├─    (|000⟩+|001⟩+|010⟩+...+|111⟩)/√8
     └───┘
     ┌───┐
q2: ─┤ H ├─
     └───┘
```

**Phase kickback (used in many algorithms):**

```
     ┌───┐
q0: ─┤ H ├──●───   Control qubit picks up phase
     └───┘  │      from the controlled operation
          ┌─┴─┐    (even though target "changes")
q1: ──────┤ U ├─
          └───┘
```

---

## Key Quantum Algorithms

### Deutsch-Jozsa Algorithm

**Problem**: Is function f(x) constant or balanced?

```
Classical: Need 2^(n-1) + 1 queries in worst case
Quantum:   Need just 1 query!

Circuit:
     ┌───┐     ┌───┐┌───┐
|0⟩ ─┤ H ├──●──┤ H ├┤ M ├─ If 0: constant
     └───┘  │  └───┘└───┘  If 1: balanced
          ┌─┴─┐
|1⟩ ─┬────┤Uf ├────────────
     │    └───┘
     H
```

### Grover's Search Algorithm

**Problem**: Find x where f(x) = 1 in unsorted database

```
Classical: O(N) queries
Quantum:   O(√N) queries  ← Quadratic speedup!

For N = 1,000,000:
  Classical: ~1,000,000 checks
  Quantum:   ~1,000 checks

The Algorithm (Grover iteration):
1. Start with equal superposition
2. Oracle: flip sign of target state
3. Diffusion: reflect about average
4. Repeat ~√N times
5. Measure → high probability of answer
```

```
Amplitude evolution:

Before:    ─────────────────  (all equal)
                 │ target

After Oracle:    ─────────────  (target flipped)
                 │
                 ▼

After Diffusion: ──────────────  (target amplified!)
                      │
                      █ target higher
```

### Shor's Algorithm

**Problem**: Factor large number N

```
Classical: Best known is sub-exponential
Quantum:   O((log N)³) ← Exponential speedup!

2048-bit RSA:
  Classical: ~10^14 years
  Quantum:   hours (with ~4000 logical qubits)

This is why quantum computers threaten current encryption!

The key insight: Finding the period of modular exponentiation
                 is naturally suited to quantum (QFT)
```

### Variational Quantum Eigensolver (VQE)

**Problem**: Find ground state energy of molecule

```
Hybrid classical-quantum:

┌─────────────────────────────────────────────────┐
│                                                  │
│   Classical         Quantum                      │
│   Computer          Computer                     │
│                                                  │
│   Optimizer         ┌───────────────────┐       │
│      │              │ Parameterized     │       │
│      │   θ          │ Circuit           │       │
│      ├─────────────▶│ U(θ)              │       │
│      │              └─────────┬─────────┘       │
│      │                        │                  │
│      │    ⟨H⟩ = E             │ Measure         │
│      ◀────────────────────────┘                  │
│                                                  │
│   Repeat until energy converges                  │
│                                                  │
└─────────────────────────────────────────────────┘

Useful TODAY on NISQ devices (noisy, limited qubits)
```

### Quantum Fourier Transform (QFT)

**The quantum equivalent of FFT:**

```
Classical FFT: O(N log N)
Quantum QFT:   O((log N)²)

Used in: Shor's algorithm, phase estimation, 
         quantum simulation
```

---

## Error Correction & Noise

### The Problem: Decoherence

```
Qubits are fragile:

     Fresh qubit         After microseconds
     
         |+⟩                  ???
          ●                    ●
                              /|\
     Clean superposition    Environmental noise
                            scrambles the state
                            
Causes:
• Thermal fluctuations
• Electromagnetic interference  
• Cosmic rays (really!)
• Any interaction with environment
```

### Error Types

| Error | Effect | Classical Equivalent |
|-------|--------|---------------------|
| **Bit flip (X)** | \|0⟩ ↔ \|1⟩ | Same as classical bit flip |
| **Phase flip (Z)** | \|+⟩ ↔ \|-⟩ | No classical analog! |
| **Bit+Phase (Y)** | Both | - |

### Quantum Error Correction

**Key insight**: Spread information across multiple physical qubits

```
3-qubit bit flip code:

Encode: |0⟩ → |000⟩
        |1⟩ → |111⟩

If one qubit flips:
|000⟩ → |100⟩  or  |010⟩  or  |001⟩

Majority vote corrects it!
(Need to do this without measuring the actual value)
```

**Surface codes** (most promising):

```
┌───┬───┬───┬───┐
│ D │ Z │ D │ Z │    D = Data qubit
├───┼───┼───┼───┤    Z = Z-syndrome measurement
│ X │ D │ X │ D │    X = X-syndrome measurement  
├───┼───┼───┼───┤
│ D │ Z │ D │ Z │    One logical qubit = many physical qubits
├───┼───┼───┼───┤
│ X │ D │ X │ D │    E.g., ~1000 physical → 1 logical
└───┴───┴───┴───┘
```

### NISQ Era

**Noisy Intermediate-Scale Quantum** (where we are now):

```
Current situation:
• 50-1000+ physical qubits
• No error correction (too expensive)
• Gate errors: ~0.1-1%
• Coherence time: microseconds to milliseconds

What this means:
• Can only run shallow circuits (~100 gates)
• Results are noisy
• Need error mitigation (not correction)
• Good for: VQE, QAOA, small simulations
```

---

## Current State of Hardware

### Major Players (2024+)

| Company | Technology | Qubits | Notes |
|---------|-----------|--------|-------|
| **IBM** | Superconducting | 1000+ | Qiskit ecosystem, cloud access |
| **Google** | Superconducting | 70+ | Demonstrated "quantum supremacy" |
| **IonQ** | Trapped ions | 30+ | High fidelity, all-to-all connectivity |
| **Quantinuum** | Trapped ions | 32+ | Highest 2-qubit gate fidelity |
| **Xanadu** | Photonic | 216+ modes | Gaussian boson sampling |
| **QuEra** | Neutral atoms | 256+ | Programmable atom arrays |

### Roadmaps

```
Timeline (approximate):

2024-2026: NISQ Era
├── 1000+ noisy qubits
├── Error mitigation techniques
├── Useful for specific chemistry/optimization
└── Quantum advantage in narrow domains

2027-2030: Early Fault-Tolerant
├── First logical qubits with error correction
├── ~100 logical qubits
├── Longer coherent computations
└── More practical applications

2030+: Full Fault-Tolerant
├── 1000+ logical qubits
├── Run Shor's algorithm on practical sizes
├── Break current encryption (RSA-2048)
└── General-purpose quantum computing
```

### Quantum Volume & Benchmarks

```
Quantum Volume: A holistic metric

QV = 2^n where n is the largest "useful" circuit

QV 64   → Can run 6x6 random circuit reliably
QV 128  → 7x7
QV 2048 → 11x11

Higher = more useful computation possible

Other metrics:
• Gate fidelity (how accurate operations are)
• Coherence time (how long qubits stay quantum)
• Connectivity (which qubits can interact)
```

---

## Cheat Sheet

### Notation Quick Reference

| Symbol | Meaning |
|--------|---------|
| \|0⟩, \|1⟩ | Basis states (kets) |
| ⟨0\|, ⟨1\| | Dual vectors (bras) |
| \|ψ⟩ | General quantum state |
| ⟨ψ\|φ⟩ | Inner product |
| \|ψ⟩⊗\|φ⟩ | Tensor product (combined system) |
| α, β | Amplitudes (complex numbers) |
| \|α\|² | Probability |

### Key Formulas

```
Superposition:     |ψ⟩ = α|0⟩ + β|1⟩,  |α|² + |β|² = 1

Measurement prob:  P(outcome) = |⟨outcome|ψ⟩|²

Multi-qubit:       |ψ⟩ = Σ αᵢ|i⟩,  Σ|αᵢ|² = 1

Hadamard:          H|0⟩ = |+⟩ = (|0⟩+|1⟩)/√2
                   H|1⟩ = |-⟩ = (|0⟩-|1⟩)/√2

Bell state:        |Φ+⟩ = (|00⟩ + |11⟩)/√2
```

### Quantum vs Classical Complexity

| Problem | Classical | Quantum |
|---------|-----------|---------|
| Unstructured search | O(N) | O(√N) Grover |
| Factoring | Sub-exponential | O((log N)³) Shor |
| Simulation | Exponential | Polynomial |
| BQP problems | Exponential | Polynomial |

---

## Key Takeaways

1. **Qubits use superposition** — exist in multiple states until measured
2. **Measurement collapses** — you only get one answer per run
3. **Entanglement enables** — correlations impossible classically
4. **Interference is key** — quantum algorithms amplify right answers
5. **Not a replacement** — quantum computers are specialized co-processors
6. **Currently NISQ** — noisy, limited, but useful for specific problems
7. **Shor threatens crypto** — post-quantum cryptography is being standardized
8. **Error correction is expensive** — ~1000 physical qubits per logical qubit

---

Next: [Practical Quantum Computing](./02_PRACTICAL.md) — Programming frameworks, simulators, and hands-on examples
