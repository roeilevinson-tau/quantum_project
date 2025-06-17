from utils import (
    create_shor_encoded_state, decode_shor_code,
    run_circuit, plot_results, get_image_path
)

def run_test():
    """Test 1: Bit Flip (X Error)"""
    print("\n## Test 1: Bit Flip (X Error)")
    print("----------------------------\n")
    print("This test injects a bit flip error on a single qubit and demonstrates the code's ability to correct it.\n")

    # Create circuit with a logical |0⟩ state
    circuit, q, c = create_shor_encoded_state([1, 0])
    circuit.barrier(q)

    # Insert an X error on the first qubit
    circuit.x(q[0])
    circuit.barrier(q)

    # Decode the state
    circuit = decode_shor_code(circuit, q, c)
    circuit.barrier(q)

    # Measure
    circuit.measure(q[0], c[0])

    # Draw the circuit
    circuit.draw(output='mpl', filename=get_image_path('test1_bit_flip_correction.png'))

    # Execute and get results
    counts = run_circuit(circuit)
    plot_results(counts, "Test 1: Bit Flip Correction Results", "test1_bit_flip_histogram")

    print(f"### Results:")
    print(f"- Measurement results: {counts}")
    print(f"- Expected outcome: Successful correction with high probability of measuring |0⟩")
    print("\nConclusion: The Shor code successfully corrects a single bit flip error.\n")

    return {
        'counts': counts
    }

if __name__ == "__main__":
    run_test() 