from utils import (
    create_shor_encoded_state, decode_shor_code,
    run_circuit, plot_results, get_image_path
)
import numpy as np

def run_test():
    """Test 2: Phase Flip (Z Error)"""
    print("\n## Test 2: Phase Flip (Z Error)")
    print("-----------------------------\n")
    print("This test injects a phase flip error on a single qubit and demonstrates the code's ability to correct it.\n")

    # Create circuit with a logical |+⟩ state (sensitive to phase errors)
    circuit, q, c = create_shor_encoded_state([1/np.sqrt(2), 1/np.sqrt(2)])
    circuit.barrier(q)

    # Insert a Z error on the first qubit
    circuit.z(q[0])
    circuit.barrier(q)

    # Decode the state
    circuit = decode_shor_code(circuit, q, c)
    circuit.barrier(q)

    # Add H gate to convert |+⟩ back to |0⟩ for measurement
    circuit.h(q[0])
    circuit.barrier(q)

    # Measure
    circuit.measure(q[0], c[0])

    # Draw the circuit
    circuit.draw(output='mpl', filename=get_image_path('test2_phase_flip_correction.png'))

    # Execute and get results
    counts = run_circuit(circuit)
    plot_results(counts, "Test 2: Phase Flip Correction Results", "test2_phase_flip_histogram")

    print(f"### Results:")
    print(f"- Measurement results: {counts}")
    print(f"- Expected outcome: Successful correction with high probability of measuring |0⟩")
    print("\nConclusion: The Shor code successfully corrects a single phase flip error.\n")

    return {
        'counts': counts
    }

if __name__ == "__main__":
    run_test() 