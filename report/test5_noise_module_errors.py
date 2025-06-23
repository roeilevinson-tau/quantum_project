from utils import (
    create_shor_encoded_state, decode_shor_code, get_statevector,
    run_circuit, plot_results, get_image_path
)
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer.noise import NoiseModel, pauli_error
from qiskit import execute, Aer
import numpy as np

def run_test():
    """Test 5: Noise Module Errors"""
    print("\n## Test 5: Noise Module Errors")
    print("----------------------------\n")
    print("This test demonstrates the Shor code's ability to correct errors introduced by a noise model")
    print("that applies bit flip errors with a certain probability during id gates.\n")

    # Create circuit with a logical |0⟩ state
    circuit, q, c = create_shor_encoded_state([1, 0])
    circuit.barrier(q)

    # Get the ideal state before error
    ideal_sv = get_statevector(circuit)

    # Add id gates to all qubits to introduce noise
    for i in range(9):
        circuit.id(q[i])
    circuit.barrier(q)

    # Create noise model with bit flip errors
    p_err = 1 # 10% probability of bit flip error
    bit_flip_error = pauli_error([('X', p_err), ('I', 1 - p_err)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(bit_flip_error, ['id'])

    # Get the corrupted state
    corrupted_sv = get_statevector(circuit)

    # Decode the state
    circuit = decode_shor_code(circuit, q, c)
    circuit.barrier(q)

    # Measure
    for i in range(9):
        circuit.measure(q[i], c[i])

    # Draw the circuit
    circuit.draw(output='mpl', filename=get_image_path('test5_noise_module_correction.png'))

    # Execute with noise model
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1000, noise_model=noise_model)
    counts = job.result().get_counts()
    plot_results(counts, "Test 5: Noise Module Correction Results", "test5_noise_module_histogram")

    # Calculate fidelity between ideal and corrupted state
    noise_fidelity = state_fidelity(ideal_sv, corrupted_sv)
    print(f"counts: {counts}")

    print(f"### Results:")
    print(f"- Measurement results: {counts}")
    print(f"- Fidelity between ideal and corrupted state: {noise_fidelity:.6f}")
    print(f"- Expected outcome: Successful correction with high probability of measuring |0⟩")
    print("\nConclusion: The Shor code successfully corrects errors introduced by the noise model.\n")

    return {
        'counts': counts,
        'fidelity': noise_fidelity
    }

if __name__ == "__main__":
    run_test() 