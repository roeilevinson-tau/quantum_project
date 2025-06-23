from utils import ShorECTest
from qiskit.quantum_info import state_fidelity

class MultipleErrorsTest(ShorECTest):
    DEFAULT_TEST_NAME = "multiple_errors"
    
    def custom_circuit_logic(self):
        """Add multiple phase flip errors to demonstrate code limitations."""

        
        self.circuit.z(self.q[0])   
        self.circuit.z(self.q[3]) 
        self.circuit.z(self.q[6])


        self.circuit.barrier(self.q)

    def run_test(self):
        """Test 4: Multiple Errors (Beyond Code's Capability)"""
        print("\n## Test 4: Multiple Errors (Beyond Code's Capability)")
        print("--------------------------------------------------\n")
        print("This test injects phase flip errors on multiple qubits to demonstrate the code's limitations.\n")
        
        # Draw the circuit
        self.draw()
        
        # Execute and get results
        counts = self.run_simulation(shots=1000)  # Ensure we have enough shots for good statistics
        

        print(f"### Results:")
        print(f"- Measurement results: {counts}")
        print(f"- Expected outcome: Failed correction due to multiple phase flip errors")
        print("\nExplanation:")
        print("1. We started with logical |0⟩ state")
        print("2. Applied Z gates (phase flip errors) to qubits 0, 3, and 6")
        print("3. These phase flips affect multiple code blocks simultaneously")
        print("4. The Shor code can only correct single errors, so multiple errors exceed its capability")
        print("5. Phase flip errors don't change measurement outcomes in computational basis directly,")
        print("   but they corrupt the quantum state's phase relationships")
        
        return {
            'counts': counts,
        }

def run_test():
    """Run the multiple errors test."""
    test = MultipleErrorsTest([1, 0])
    return test.run_test()

if __name__ == "__main__":
    run_test() 