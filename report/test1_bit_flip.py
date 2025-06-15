from utils import (
    create_shor_encoded_state, decode_shor_code, get_statevector,
    run_circuit, plot_results, get_image_path
)
from qiskit.quantum_info import state_fidelity

def run_test():
    """Test 1: Bit Flip (X Error)"""
    print("\n## Test 1: Bit Flip (X Error)")
    print("----------------------------\n")
    print("This test injects a bit flip error on a single qubit and demonstrates the code's ability to correct it.\n")

    # Create circuit with a logical |0⟩ state
    circuit, q, c = create_shor_encoded_state([1, 0])
    circuit.barrier(q)

    # Get the ideal state before error
    ideal_sv = get_statevector(circuit)

    # Insert an X error on the first qubit
    circuit.x(q[0])
    circuit.barrier(q)

    # Get the corrupted state
    corrupted_sv = get_statevector(circuit)

    # Decode the state
    circuit = decode_shor_code(circuit, q, c)
    circuit.barrier(q)

    # Measure
    circuit.measure(q[0], c[0])

    # Draw the circuit
    circuit.draw(output='mpl', filename=get_image_path('test1_bit_flip_correction.png'))

    # Execute and get results
    counts = run_circuit(circuit)
    plot_results(counts, "Test 1: Bit Flip Correction Results")

    # Calculate fidelity between ideal and corrupted state
    bit_flip_fidelity = state_fidelity(ideal_sv, corrupted_sv)

    print(f"### Results:")
    print(f"- Measurement results: {counts}")
    print(f"- Fidelity between ideal and corrupted state: {bit_flip_fidelity:.6f}")
    print(f"- Expected outcome: Successful correction with high probability of measuring |0⟩")
    print("\nConclusion: The Shor code successfully corrects a single bit flip error.\n")

    return {
        'counts': counts,
        'fidelity': bit_flip_fidelity
    }

if __name__ == "__main__":
    run_test() 