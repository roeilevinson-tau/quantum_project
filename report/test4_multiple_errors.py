from utils import (
    create_shor_encoded_state, decode_shor_code, get_statevector,
    run_circuit, plot_results, get_image_path
)
from qiskit.quantum_info import state_fidelity

def run_test():
    """Test 4: Multiple Errors (Beyond Code's Capability)"""
    print("\n## Test 4: Multiple Errors (Beyond Code's Capability)")
    print("--------------------------------------------------\n")
    print("This test injects errors on multiple qubits to demonstrate the code's limitations.\n")

    # Create circuit with a logical |0⟩ state
    circuit, q, c = create_shor_encoded_state([1, 0])
    circuit.barrier(q)

    # Get the ideal state before error
    ideal_sv = get_statevector(circuit)

    # Insert two errors in the first block
    # In the first block (qubits 0,1,2), we flip qubits 0 and 1
    # This means 2 qubits vote for |1⟩ and 1 qubit votes for |0⟩
    # The majority voting will incorrectly decide the block is in state |1⟩
    print("Applying X errors to qubits 0 and 1 in the first block")
    circuit.x(q[0])
    circuit.x(q[1])
    circuit.barrier(q)

    # Get the corrupted state
    corrupted_sv = get_statevector(circuit)

    # Decode the state - this should fail due to incorrect majority voting
    circuit = decode_shor_code(circuit, q, c)
    circuit.barrier(q)

    # Measure
    circuit.measure(q[0], c[0])

    # Draw the circuit
    circuit.draw(output='mpl', filename=get_image_path('test4_multiple_errors.png'))

    # Execute and get results
    counts = run_circuit(circuit, shots=1000)  # Ensure we have enough shots for good statistics
    plot_results(counts, "Test 4: Multiple Errors Results", "test4_multiple_errors_histogram")

    # Calculate fidelity between ideal and corrupted state
    multiple_errors_fidelity = state_fidelity(ideal_sv, corrupted_sv)

    print(f"### Results:")
    print(f"- Measurement results: {counts}")
    print(f"- Fidelity between ideal and corrupted state: {multiple_errors_fidelity:.6f}")
    print(f"- Expected outcome: Failed correction with high probability of measuring |1⟩")
    print("\nExplanation:")
    print("1. We started with logical |0⟩ state")
    print("2. Applied X errors to qubits 0 and 1 in the first block")
    print("3. This causes majority voting to fail because 2 qubits vote for |1⟩ and 1 for |0⟩")
    print("4. The error correction incorrectly 'corrects' to |1⟩ instead of |0⟩")
    print("\nThis demonstrates a fundamental limitation of the Shor code:")
    print("It cannot correct multiple bit flips within the same 3-qubit block")

    return {
        'counts': counts,
        'fidelity': multiple_errors_fidelity
    }

if __name__ == "__main__":
    run_test() 