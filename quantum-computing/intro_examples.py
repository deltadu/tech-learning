"""
Quantum Computing Intro Examples (Qiskit)

Run all examples: python intro_examples.py
Run specific example: python intro_examples.py <number>

Requirements:
    pip install qiskit qiskit-aer

All examples run on a classical simulator — no quantum hardware needed!
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np


def example_1_first_qubit():
    """Create a qubit in superposition and measure it."""
    print("\n" + "=" * 60)
    print("Example 1: Your First Qubit")
    print("=" * 60)

    # Create circuit: 1 qubit, 1 classical bit
    qc = QuantumCircuit(1, 1)

    # Put qubit in superposition (|0⟩ + |1⟩)/√2
    qc.h(0)

    # Measure the qubit
    qc.measure(0, 0)

    print("\nCircuit:")
    print(qc.draw())

    # Run 1000 times on simulator
    sim = AerSimulator()
    result = sim.run(qc, shots=1000).result()
    counts = result.get_counts()

    print(f"\nResults (1000 shots): {counts}")
    print("→ Roughly 50/50 split — that's superposition!")


def example_2_visualize_circuit():
    """Build and visualize a simple circuit."""
    print("\n" + "=" * 60)
    print("Example 2: Visualize a Circuit")
    print("=" * 60)

    qc = QuantumCircuit(2)
    qc.h(0)  # Hadamard on qubit 0
    qc.cx(0, 1)  # CNOT: entangle qubits

    print("\nCircuit diagram:")
    print(qc.draw())
    print("\nReading left-to-right:")
    print("  1. H gate puts qubit 0 in superposition")
    print("  2. CNOT flips qubit 1 if qubit 0 is |1⟩")
    print("  → Result: entangled Bell state!")


def example_3_bell_state():
    """Create and measure an entangled Bell state."""
    print("\n" + "=" * 60)
    print("Example 3: Bell State (Entanglement)")
    print("=" * 60)

    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Superposition on qubit 0
    qc.cx(0, 1)  # Entangle with qubit 1
    qc.measure([0, 1], [0, 1])

    print("\nCircuit:")
    print(qc.draw())

    sim = AerSimulator()
    result = sim.run(qc, shots=1000).result()
    counts = result.get_counts()

    print(f"\nResults: {counts}")
    print("→ Only '00' and '11' — never '01' or '10'!")
    print("→ Qubits are perfectly correlated (entangled)")


def example_4_statevector():
    """Inspect the quantum state without measuring."""
    print("\n" + "=" * 60)
    print("Example 4: See the Quantum State")
    print("=" * 60)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    print("\nCircuit:")
    print(qc.draw())

    # Get exact state vector (only possible in simulation!)
    state = Statevector(qc)

    print(f"\nState vector: {state.data}")
    print("\nInterpretation:")
    print("  [0.707, 0, 0, 0.707] means:")
    print("  • 0.707 amplitude for |00⟩")
    print("  • 0 amplitude for |01⟩")
    print("  • 0 amplitude for |10⟩")
    print("  • 0.707 amplitude for |11⟩")

    print(f"\nProbabilities: {state.probabilities()}")
    print("  → 50% chance of |00⟩, 50% chance of |11⟩")


def example_5_gates():
    """Apply different quantum gates and see their effects."""
    print("\n" + "=" * 60)
    print("Example 5: Different Gates")
    print("=" * 60)

    # Start with |0⟩
    qc = QuantumCircuit(1)
    print(f"Start |0⟩: {Statevector(qc).data}")

    # X gate (NOT): |0⟩ → |1⟩
    qc.x(0)
    print(f"After X:   {Statevector(qc).data}  (flipped to |1⟩)")

    # H gate: |1⟩ → |-⟩ = (|0⟩ - |1⟩)/√2
    qc.h(0)
    state_after_h = Statevector(qc).data
    print(f"After H:   {state_after_h}  (superposition with negative phase)")

    # Rotation around Y
    qc.ry(np.pi / 4, 0)
    print(f"After Ry:  {Statevector(qc).data}  (rotated 45° around Y)")

    print("\nFull circuit:")
    print(qc.draw())


def example_6_measurement_bases():
    """Measure the same state in different bases."""
    print("\n" + "=" * 60)
    print("Example 6: Measurement Bases")
    print("=" * 60)

    sim = AerSimulator()

    # Prepare |+⟩ state
    print("Preparing |+⟩ = (|0⟩ + |1⟩)/√2")

    # Measure in Z basis (computational basis)
    qc_z = QuantumCircuit(1, 1)
    qc_z.h(0)  # Create |+⟩
    qc_z.measure(0, 0)

    result_z = sim.run(qc_z, shots=1000).result().get_counts()
    print(f"\nZ-basis measurement: {result_z}")
    print("  → Random! |+⟩ has equal amplitudes for |0⟩ and |1⟩")

    # Measure in X basis (apply H to convert X-basis to Z-basis)
    qc_x = QuantumCircuit(1, 1)
    qc_x.h(0)  # Create |+⟩
    qc_x.h(
        0
    )  # Convert to Z-basis (H·H = I, but |+⟩ in X-basis = |0⟩ in Z-basis)
    qc_x.measure(0, 0)

    result_x = sim.run(qc_x, shots=1000).result().get_counts()
    print(f"\nX-basis measurement: {result_x}")
    print("  → Deterministic! |+⟩ is an eigenstate of X")


def example_7_grover_search():
    """Simple 2-qubit Grover's search algorithm."""
    print("\n" + "=" * 60)
    print("Example 7: Grover's Search (2 qubits)")
    print("=" * 60)

    print("\nSearching for |11⟩ among 4 possible states...")

    qc = QuantumCircuit(2, 2)

    # 1. Create uniform superposition
    qc.h([0, 1])
    qc.barrier()  # Visual separator

    # 2. Oracle: mark |11⟩ with phase flip
    # CZ flips phase only when both qubits are |1⟩
    qc.cz(0, 1)
    qc.barrier()

    # 3. Diffusion operator (reflect about mean)
    qc.h([0, 1])
    qc.z([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1])
    qc.barrier()

    # 4. Measure
    qc.measure([0, 1], [0, 1])

    print("\nCircuit:")
    print(qc.draw())

    sim = AerSimulator()
    result = sim.run(qc, shots=1000).result()
    counts = result.get_counts()

    print(f"\nResults: {counts}")
    print("→ Found |11⟩ with ~100% probability!")
    print(
        "→ Classical search needs ~2 tries on average, Grover finds it in 1 iteration"
    )


def example_8_multi_qubit():
    """Work with multiple qubits."""
    print("\n" + "=" * 60)
    print("Example 8: Multi-Qubit Operations")
    print("=" * 60)

    # 3-qubit GHZ state: (|000⟩ + |111⟩)/√2
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    print("\nCreating GHZ state (|000⟩ + |111⟩)/√2:")
    print(qc.draw())

    sim = AerSimulator()
    result = sim.run(qc, shots=1000).result()
    counts = result.get_counts()

    print(f"\nResults: {counts}")
    print("→ All 3 qubits perfectly correlated!")


def common_gates_reference():
    """Quick reference for common gates."""
    print("\n" + "=" * 60)
    print("Quick Reference: Common Gates")
    print("=" * 60)

    qc = QuantumCircuit(3)

    print("\nSingle-qubit gates:")
    print("  qc.h(0)        # Hadamard: |0⟩ → |+⟩")
    print("  qc.x(0)        # Pauli-X (NOT): |0⟩ ↔ |1⟩")
    print("  qc.y(0)        # Pauli-Y")
    print("  qc.z(0)        # Pauli-Z: |1⟩ → -|1⟩")
    print("  qc.s(0)        # S gate (√Z)")
    print("  qc.t(0)        # T gate (√S)")

    print("\nRotation gates (angle in radians):")
    print("  qc.rx(θ, 0)    # Rotate around X-axis")
    print("  qc.ry(θ, 0)    # Rotate around Y-axis")
    print("  qc.rz(θ, 0)    # Rotate around Z-axis")

    print("\nTwo-qubit gates:")
    print("  qc.cx(0, 1)    # CNOT: flip qubit 1 if qubit 0 is |1⟩")
    print("  qc.cz(0, 1)    # CZ: phase flip if both |1⟩")
    print("  qc.swap(0, 1)  # Swap two qubits")
    print("  qc.cp(θ, 0, 1) # Controlled phase rotation")

    print("\nThree-qubit gates:")
    print("  qc.ccx(0, 1, 2)   # Toffoli (CCX): flip if both controls |1⟩")
    print("  qc.cswap(0, 1, 2) # Fredkin: swap if control is |1⟩")


def run_all():
    """Run all examples."""
    example_1_first_qubit()
    example_2_visualize_circuit()
    example_3_bell_state()
    example_4_statevector()
    example_5_gates()
    example_6_measurement_bases()
    example_7_grover_search()
    example_8_multi_qubit()
    common_gates_reference()

    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Try modifying the circuits")
    print("  • Run on real hardware: https://quantum-computing.ibm.com")
    print("  • See 02_PRACTICAL.md for more advanced topics")


if __name__ == "__main__":
    import sys

    examples = {
        "1": example_1_first_qubit,
        "2": example_2_visualize_circuit,
        "3": example_3_bell_state,
        "4": example_4_statevector,
        "5": example_5_gates,
        "6": example_6_measurement_bases,
        "7": example_7_grover_search,
        "8": example_8_multi_qubit,
        "ref": common_gates_reference,
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            examples[choice]()
        else:
            print(f"Unknown example: {choice}")
            print(f"Available: {list(examples.keys())}")
    else:
        run_all()
