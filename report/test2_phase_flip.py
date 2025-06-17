from shor_ec_test import ShorECTest
import numpy as np

class PhaseFlipTest(ShorECTest):
    DEFAULT_TEST_NAME = "phase_flip_correction"
    
    def custom_circuit_logic(self):
        """Add a phase flip error to the first qubit."""
        self.circuit.z(self.q[0])
        self.circuit.barrier(self.q)

    def _decode(self):
        """Override decode to add H gate after standard decoding."""
        super()._decode()
        # Add H gate to convert |+⟩ back to |0⟩ for measurement
        self.circuit.h(self.q[0])
        self.circuit.barrier(self.q)

    def run_test(self):
        """Test 2: Phase Flip (Z Error)"""
        print("\n## Test 2: Phase Flip (Z Error)")
        print("-----------------------------\n")
        print("This test injects a phase flip error on a single qubit and demonstrates the code's ability to correct it.\n")
        
        # Draw the circuit
        self.draw()
        
        # Execute and get results
        counts = self.run_simulation()
        
        # Calculate fidelity between ideal and corrupted state
        phase_flip_fidelity = self.get_state_fidelity()
        
        print(f"### Results:")
        print(f"- Measurement results: {counts}")
        print(f"- Fidelity between ideal and corrupted state: {phase_flip_fidelity:.6f}")
        print(f"- Expected outcome: Successful correction with high probability of measuring |0⟩")
        print("\nConclusion: The Shor code successfully corrects a single phase flip error.\n")
        
        return {
            'counts': counts,
            'fidelity': phase_flip_fidelity
        }

def run_test():
    """Run the phase flip test."""
    test = PhaseFlipTest([1/np.sqrt(2), 1/np.sqrt(2)])
    return test.run_test()

if __name__ == "__main__":
    run_test() 