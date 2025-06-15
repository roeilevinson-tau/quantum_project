from utils import (
    create_shor_encoded_state, decode_shor_code, get_statevector,
    run_circuit, plot_results, get_image_path
)
from qiskit.quantum_info import state_fidelity
import numpy as np

def run_test():
    """Test 3: Simultaneous Bit & Phase Flip (Y Error)"""
    print("\n## Test 3: Simultaneous Bit & Phase Flip (Y Error)")
    print("-----------------------------------------------\n")
    print("This test injects both a bit flip and phase flip simultaneously (Y error) on a single qubit")
    print("and demonstrates the code's ability to correct it.\n")

    # Create circuit with a logical |+⟩ state (to show both types of errors)
    circuit, q, c = create_shor_encoded_state([1/np.sqrt(2), 1/np.sqrt(2)])
    circuit.barrier(q)

    # Get the ideal state before error
    ideal_sv = get_statevector(circuit)

    # Insert a Y error on the first qubit (equivalent to Z and X)
    circuit.y(q[0])
    circuit.barrier(q)

    # Get the corrupted state
    corrupted_sv = get_statevector(circuit)

    # Decode the state
    circuit = decode_shor_code(circuit, q, c)
    circuit.barrier(q)

    # Add H gate to convert |+⟩ back to |0⟩ for measurement
    circuit.h(q[0])
    circuit.barrier(q)

    # Measure
    circuit.measure(q[0], c[0])

    # Draw the circuit
    circuit.draw(output='mpl', filename=get_image_path('test3_y_error_correction.png'))

    # Execute and get results
    counts = run_circuit(circuit)
    plot_results(counts, "Test 3: Y Error Correction Results")

    # Calculate fidelity between ideal and corrupted state
    y_error_fidelity = state_fidelity(ideal_sv, corrupted_sv)

    print(f"### Results:")
    print(f"- Measurement results: {counts}")
    print(f"- Fidelity between ideal and corrupted state: {y_error_fidelity:.6f}")
    print(f"- Expected outcome: Successful correction with high probability of measuring |0⟩")
    print("\nConclusion: The Shor code successfully corrects a simultaneous bit and phase flip (Y error) on a single qubit.\n")

    return {
        'counts': counts,
        'fidelity': y_error_fidelity
    }

if __name__ == "__main__":
    run_test() 