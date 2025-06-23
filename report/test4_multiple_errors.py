from shor_ec_test import ShorECTest

class MultipleErrorsTest(ShorECTest):
    DEFAULT_TEST_NAME = "multiple_errors"
    
    def custom_circuit_logic(self):
        """Add multiple bit flip errors to demonstrate code limitations."""
        # Insert three bit flip errors in the first block
        # This will definitely break the majority voting
        print("Applying X errors to all three qubits in the first block")
        self.circuit.x(self.q[0])  # Flip first qubit
        self.circuit.x(self.q[1])  # Flip second qubit
        self.circuit.x(self.q[2])  # Flip third qubit - this makes all qubits in the block flipped
        self.circuit.barrier(self.q)

    def run_test(self):
        """Test 4: Multiple Errors (Beyond Code's Capability)"""
        print("\n## Test 4: Multiple Errors (Beyond Code's Capability)")
        print("--------------------------------------------------\n")
        print("This test injects errors on multiple qubits to demonstrate the code's limitations.\n")
        
        # Draw the circuit
        self.draw()
        
        # Execute and get results
        counts = self.run_simulation(shots=1000)  # Ensure we have enough shots for good statistics
        

        print(f"### Results:")
        print(f"- Measurement results: {counts}")
        print(f"- Expected outcome: Failed correction with high probability of measuring |1⟩")
        print("\nExplanation:")
        print("1. We started with logical |0⟩ state")
        print("2. Applied X errors to ALL qubits in the first block (qubits 0, 1, and 2)")
        print("3. This completely inverts the state of the first block")
        print("4. The majority voting will see all qubits as |1⟩ and incorrectly assume this is the correct state")
        
        return {
            'counts': counts,
        }

def run_test():
    """Run the multiple errors test."""
    test = MultipleErrorsTest([1, 0], should_ec_fail=True)
    return test.run_test()

if __name__ == "__main__":
    run_test() 