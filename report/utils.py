from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import state_fidelity, Statevector
import os

# Get the backend simulator
aer_sim = Aer.get_backend('aer_simulator')
sv_sim = Aer.get_backend('statevector_simulator')

class ShorECTest:
    DEFAULT_TEST_NAME = "override_this_string"
    
    def __init__(self, initial_state=[1, 0], test_name=None):
        """
        Initialize the error correction test circuit.
        
        Args:
            initial_state (list): List of 2 complex numbers for the |0⟩ and |1⟩ amplitudes
            test_name (str): Name of the test for file naming
        """
        self._test_name = test_name or self.DEFAULT_TEST_NAME
        self._initial_state = initial_state
        
        # Create quantum and classical registers
        self.q = QuantumRegister(9, 'q')
        self.c = ClassicalRegister(9, 'c')
        self.circuit = QuantumCircuit(self.q, self.c)
        
        # Build the circuit
        self._build_circuit()
    
    def _build_circuit(self):
        """Build the complete circuit with encoding, custom logic, and decoding."""
        # Normalize the initial state
        norm = np.sqrt(abs(self._initial_state[0])**2 + abs(self._initial_state[1])**2)
        initial_state = [self._initial_state[0]/norm, self._initial_state[1]/norm]
        
        # Initialize the first qubit to our chosen state
        if initial_state[1] != 0:
            self.circuit.initialize(initial_state, self.q[0])
        
        self._encode()
        
        # Allow for circuit modification
        self.custom_circuit_logic()
        
        # Decode
        self._decode()
        
        # Measure all qubits
        for i in range(9):
            self.circuit.measure(self.q[i], self.c[i])
    
    def custom_circuit_logic(self):
        """Override this method to modify the circuit before decoding."""
        pass

    def _encode(self):
        """Internal method to encode the Shor code."""
        # Encoding
        self.circuit.cx(self.q[0], self.q[3])
        self.circuit.cx(self.q[0], self.q[6])
        
        self.circuit.h(self.q[0])
        self.circuit.h(self.q[3])
        self.circuit.h(self.q[6])
        
        self.circuit.cx(self.q[0], self.q[1])
        self.circuit.cx(self.q[3], self.q[4])
        self.circuit.cx(self.q[6], self.q[7])
        
        self.circuit.cx(self.q[0], self.q[2])
        self.circuit.cx(self.q[3], self.q[5])
        self.circuit.cx(self.q[6], self.q[8])
        
        self.circuit.barrier(self.q)
        
        self._initial_state = Statevector.from_instruction(self.circuit)

    def _decode(self):
        """Internal method to decode the Shor code."""
        # First level of error correction - correct bit flips within each block
        # Block 1
        self.circuit.cx(self.q[0], self.q[1])
        self.circuit.cx(self.q[0], self.q[2])
        self.circuit.ccx(self.q[1], self.q[2], self.q[0])
        
        # Block 2
        self.circuit.cx(self.q[3], self.q[4])
        self.circuit.cx(self.q[3], self.q[5])
        self.circuit.ccx(self.q[4], self.q[5], self.q[3])
        
        # Block 3
        self.circuit.cx(self.q[6], self.q[7])
        self.circuit.cx(self.q[6], self.q[8])
        self.circuit.ccx(self.q[7], self.q[8], self.q[6])
        
        # Convert to phase basis for phase error correction
        self.circuit.h(self.q[0])
        self.circuit.h(self.q[3])
        self.circuit.h(self.q[6])
        
        # Second level - correct phase flips
        self.circuit.cx(self.q[0], self.q[3])
        self.circuit.cx(self.q[0], self.q[6])
        self.circuit.ccx(self.q[3], self.q[6], self.q[0])
        
        self.circuit.barrier(self.q)

        self._final_state = Statevector.from_instruction(self.circuit)
    
    def draw(self):
        """Draw the circuit and save it to a file."""
        self.circuit.draw(output='mpl', filename=get_image_path(f'{self._test_name}_circuit.png'))
    
    def run_simulation(self, shots=1000):
        """Run the circuit simulation and return the counts."""
        job = execute(self.circuit, aer_sim, shots=shots)
        job_monitor(job)
        counts = job.result().get_counts()
        self.plot_results(counts, f"{self._test_name} Results", f"{self._test_name}_histogram")
        return counts
    
    def get_statevector(self):
        """Get the statevector of the circuit."""
        return Statevector.from_instruction(self.circuit)

    def get_state_fidelity(self):
        """Get the state fidelity of the circuit."""
        if self._initial_state is None or self._final_state is None:
            raise ValueError("Statevectors not initialized. Run the circuit first.")
        return state_fidelity(self._initial_state, self._final_state)

    def plot_results(self, results_dict, title, filename=None):
        """Visualize measurement results."""
        labels = list(results_dict.keys())
        values = list(results_dict.values())
        
        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)
        plt.title(title)
        plt.ylabel('Counts')
        plt.xlabel('Measurement Outcome')
        if filename is None:
            filename = title.replace(' ', '_')
        plt.savefig(get_image_path(f"{filename}.png"))
        plt.close()

# Utility functions
def ensure_images_dir():
    """Create the images directory if it doesn't exist."""
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir

def get_image_path(filename):
    """Get the full path for an image file."""
    return os.path.join(ensure_images_dir(), filename) 