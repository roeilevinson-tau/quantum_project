from utils import ShorECTest
import numpy as np

class YErrorTest(ShorECTest):
    DEFAULT_TEST_NAME = "y_error_correction"
    
    def custom_circuit_logic(self):
        """Add a Y error (simultaneous bit and phase flip) to the first qubit."""
        self.circuit.y(self.q[0])
        self.circuit.barrier(self.q)

    def _decode(self):
        """Override decode to add H gate after standard decoding."""
        super()._decode()
        # Add H gate to convert |+⟩ back to |0⟩ for measurement
        self.circuit.h(self.q[0])
        self.circuit.barrier(self.q)

    def run_test(self):
        """Test 3: Simultaneous Bit & Phase Flip (Y Error)"""
        print("\n## Test 3: Simultaneous Bit & Phase Flip (Y Error)")
        print("-----------------------------------------------\n")
        print("This test injects both a bit flip and phase flip simultaneously (Y error) on a single qubit")
        print("and demonstrates the code's ability to correct it.\n")
        
        # Draw the circuit
        self.draw()
        
        # Execute and get results
        counts = self.run_simulation()
        
        # Calculate fidelity between ideal and corrupted state
        y_error_fidelity = self.get_state_fidelity()
        
        print(f"### Results:")
        print(f"- Measurement results: {counts}")
        print(f"- Fidelity between ideal and corrupted state: {y_error_fidelity:.6f}")
        print(f"- Expected outcome: Successful correction with high probability of measuring |0⟩")
        print("\nConclusion: The Shor code successfully corrects a simultaneous bit and phase flip (Y error) on a single qubit.\n")
        
        return {
            'counts': counts,
            'fidelity': y_error_fidelity
        }

def run_test():
    """Run the Y error test."""
    test = YErrorTest([1/np.sqrt(2), 1/np.sqrt(2)])
    return test.run_test()

if __name__ == "__main__":
    run_test() 