from utils import (
    create_shor_encoded_state, decode_shor_code,
    run_circuit, plot_results, get_image_path
)

def run_test():
    """Test 4: Multiple Errors (Beyond Code's Capability)"""
    print("\n## Test 4: Multiple Errors (Beyond Code's Capability)")
    print("--------------------------------------------------\n")
    print("This test injects errors on multiple qubits to demonstrate the code's limitations.\n")

    # Create circuit with a logical |0⟩ state
    circuit, q, c = create_shor_encoded_state([1, 0])
    circuit.barrier(q)

    # Insert three bit flip errors in the first block
    # This will definitely break the majority voting
    print("Applying X errors to all three qubits in the first block")
    circuit.x(q[0])  # Flip first qubit
    circuit.x(q[1])  # Flip second qubit
    circuit.x(q[2])  # Flip third qubit - this makes all qubits in the block flipped
    circuit.barrier(q)

    # Decode the state - this should fail as all qubits in the block are flipped
    circuit = decode_shor_code(circuit, q, c)
    circuit.barrier(q)

    # Measure
    circuit.measure(q[0], c[0])

    # Draw the circuit
    circuit.draw(output='mpl', filename=get_image_path('test4_multiple_errors.png'))

    # Execute and get results
    counts = run_circuit(circuit, shots=1000)  # Ensure we have enough shots for good statistics
    plot_results(counts, "Test 4: Multiple Errors Results", "test4_multiple_errors_histogram")

    print(f"### Results:")
    print(f"- Measurement results: {counts}")
    print(f"- Expected outcome: Failed correction with high probability of measuring |1⟩")
    print("\nExplanation:")
    print("1. We started with logical |0⟩ state")
    print("2. Applied X errors to ALL qubits in the first block (qubits 0, 1, and 2)")
    print("3. This completely inverts the state of the first block")
    print("4. The majority voting will see all qubits as |1⟩ and incorrectly assume this is the correct state")

    return {
        'counts': counts
    }

if __name__ == "__main__":
    run_test() 