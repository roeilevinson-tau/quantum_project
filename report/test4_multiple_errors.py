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

    # Insert errors on two qubits
    circuit.x(q[0])
    circuit.x(q[3])
    circuit.barrier(q)

    # Get the corrupted state
    corrupted_sv = get_statevector(circuit)

    # Decode the state
    circuit = decode_shor_code(circuit, q, c)
    circuit.barrier(q)

    # Measure
    circuit.measure(q[0], c[0])

    # Draw the circuit
    circuit.draw(output='mpl', filename=get_image_path('test4_multiple_errors.png'))

    # Execute and get results
    counts = run_circuit(circuit)
    plot_results(counts, "Test 4: Multiple Errors Results")

    # Calculate fidelity between ideal and corrupted state
    multiple_errors_fidelity = state_fidelity(ideal_sv, corrupted_sv)

    print(f"### Results:")
    print(f"- Measurement results: {counts}")
    print(f"- Fidelity between ideal and corrupted state: {multiple_errors_fidelity:.6f}")
    print(f"- Expected outcome: Failed correction with significant probability of measuring |1⟩")
    print("\nConclusion: The Shor code fails when multiple errors occur beyond its correction capability.\n")
    print("The Shor code can correct at most one error of each type (bit flip and phase flip). ")
    print("When errors occur on multiple qubits, the correction breaks down.\n")

    return {
        'counts': counts,
        'fidelity': multiple_errors_fidelity
    }

if __name__ == "__main__":
    run_test() 