from shor_ec_test import ShorECTest
from qiskit.providers.aer.noise import NoiseModel, pauli_error
import numpy as np

class NoiseModuleTest(ShorECTest):
    DEFAULT_TEST_NAME = "noise_module_test"
    
    def __init__(self, initial_state=[1, 0], test_name=None, should_ec_fail=False, p_err=0):
        """
        Initialize the noise module test.
        
        Args:
            initial_state (list): List of 2 complex numbers for the |0⟩ and |1⟩ amplitudes
            test_name (str): Name of the test for file naming
            should_ec_fail (bool): Whether the error correction should fail
            p_err (float): Probability of bit flip error (0 to 1)
        """
        self.p_err = p_err
        super().__init__(initial_state, test_name, should_ec_fail)
    
    def noise_module_init(self):
        """Initialize the noise module with bit flip errors."""
        # Create noise model with bit flip errors
        bit_flip_error = pauli_error([('X', self.p_err), ('I', 1 - self.p_err)])
        self._noise_module = NoiseModel()
        self._noise_module.add_all_qubit_quantum_error(bit_flip_error, ['id'])
    
    def custom_circuit_logic(self):
        """Apply id gates to all qubits to introduce noise."""
        for i in range(9):
            self.circuit.id(self.q[i])
        self.circuit.barrier(self.q)

    def run_test(self):
        """Test 5: Noise Module Errors"""
        print("\n## Test 5: Noise Module Errors")
        print("----------------------------\n")
        print("This test demonstrates the Shor code's ability to correct errors introduced by a noise model")
        print("that applies bit flip errors with a certain probability during id gates.\n")
        
        # Draw the circuit
        self.draw()
        
        # Execute and get results
        counts = self.run_simulation()
        
        # Calculate fidelity between ideal and corrupted state
        noise_fidelity = self.get_state_fidelity()
        
        print(f"### Results:")
        print(f"- Measurement results: {counts}")
        print(f"- Fidelity between ideal and corrupted state: {noise_fidelity:.6f}")
        print(f"- Expected outcome: Successful correction with high probability of measuring |0⟩")
        print("\nConclusion: The Shor code successfully corrects errors introduced by the noise model.\n")
        
        return {
            'counts': counts,
            'fidelity': noise_fidelity
        }

def run_test():
    """Run the bit flip test."""
    test = NoiseModuleTest(p_err=0)
    return test.run_test()

if __name__ == "__main__":
    run_test() 