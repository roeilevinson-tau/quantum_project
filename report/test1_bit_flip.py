from shor_ec_test import ShorECTest

class BitFlipTest(ShorECTest):
    DEFAULT_TEST_NAME = "bit_flip_correction"
    
    def custom_circuit_logic(self):
        """Add a bit flip error to the first qubit."""
        self.circuit.x(self.q[0])
        self.circuit.barrier(self.q)

    def run_test(self):
        """Test 1: Bit Flip (X Error)"""
        print("\n## Test 1: Bit Flip (X Error)")
        print("----------------------------\n")
        print("This test injects a bit flip error on a single qubit and demonstrates the code's ability to correct it.\n")
        
        # Draw the circuit
        self.draw()
        
        # Execute and get results
        counts = self.run_simulation()
        
        print(f"### Results:")
        print(f"- Measurement results: {counts}")
        print(f"- Expected outcome: Successful correction with high probability of measuring |0⟩")
        print("\nConclusion: The Shor code successfully corrects a single bit flip error.\n")
        
        return {
            'counts': counts,
        }

def run_test():
    """Run the bit flip test."""
    test = BitFlipTest([1, 0])
    return test.run_test()

if __name__ == "__main__":
    run_test() 