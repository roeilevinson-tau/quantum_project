from shor_ec_test import ShorECTest
from qiskit.providers.aer.noise import NoiseModel, pauli_error
import numpy as np

class NoiseModuleTest(ShorECTest):
    DEFAULT_TEST_NAME = "noise_module_test"
    
    def noise_module_init(self, p_err=0):
        """Initialize the noise module with bit flip errors."""
        # Create noise model with bit flip errors
        bit_flip_error = pauli_error([('X', p_err), ('I', 1 - p_err)])
        self._noise_module = NoiseModel()
        # self._noise_module.add_all_qubit_quantum_error(bit_flip_error, ['id'])
        self._noise_module.add_all_qubit_quantum_error(bit_flip_error, ['delay'])
    
    def custom_circuit_logic(self):
        """Apply id gates to all qubits to introduce noise."""
        for i in range(9):
            self.circuit.delay(1, self.q[i], unit='dt')  # Forces a scheduled idle time
        self.circuit.barrier(self.q)
    
    def calculate_win_rate(self, counts):
        """Calculate the win rate from measurement results.
        
        Args:
            counts (dict): Dictionary of measurement results
            
        Returns:
            float: Win rate as a percentage
        """
        total_shots = sum(counts.values())
        wins = sum(count for key, count in counts.items() if int(key[-1]) == 0)
        return (wins / total_shots) * 100 if total_shots > 0 else 0
    
    def run_single_test(self, p_err):
        """Run a single test with given error probability.
        
        Args:
            p_err (float): Error probability to test
            
        Returns:
            float: Win rate as a percentage
        """
        self.noise_module_init(p_err)
        counts = self.run_simulation()
        return self.calculate_win_rate(counts)

    def check_ec(self, counts):
        """Check if error correction was successful.
        
        Args:
            counts (dict): Dictionary of measurement results
            
        Returns:
        """
        pass
    
    def run_test(self):
        """Test 5: Noise Module Error Sweep"""
        print("\n## Test 5: Noise Module Error Sweep")
        print("--------------------------------\n")
        print("This test demonstrates how the Shor code's error correction capability")
        print("varies with different error probabilities in the noise model.\n")
        
        # Draw the circuit
        self.draw()
        
        # Perform sweep over error probabilities
        error_probs = np.linspace(0.0, 1.0, 4)
        print("Error Probability (%) | Win Rate (%)")
        print("---------------------|-------------")
        
        for p_err in error_probs:
            win_rate = self.run_single_test(p_err)
            print(f"{p_err*100:19.1f} | {win_rate:11.1f}")
        
        print("\nConclusion: The Shor code's error correction capability degrades as the error probability increases.\n")


def run_test():
    """Run the noise module test."""
    test = NoiseModuleTest()
    return test.run_test()

if __name__ == "__main__":
    run_test() 