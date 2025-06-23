from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer, transpile
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import state_fidelity, Statevector
import os
from colorama import Fore, Style

# Get the backend simulator
aer_sim = Aer.get_backend('aer_simulator')
sv_sim = Aer.get_backend('statevector_simulator')


class ShorECTest:
    DEFAULT_TEST_NAME = "override_this_string"
    
    def __init__(self, initial_state=[1, 0], test_name=None, should_ec_fail=False):
        """
        Initialize the error correction test circuit.
        
        Args:
            initial_state (list): List of 2 complex numbers for the |0⟩ and |1⟩ amplitudes
            test_name (str): Name of the test for file naming
            should_ec_fail (bool): Whether the error correction should fail
        """
        self._test_name = test_name or self.DEFAULT_TEST_NAME
        self._initial_state = initial_state
        self._should_ec_fail = should_ec_fail
        
        # Create quantum and classical registers
        self.q = QuantumRegister(9, 'q')
        self.c = ClassicalRegister(9, 'c')
        self.circuit = QuantumCircuit(self.q, self.c)
        
        # Initialize noise module
        self._noise_module = None
        self.noise_module_init()
        
        # Build the circuit
        self._build_circuit()
    
    def noise_module_init(self, p_err=None):
        """Initialize the noise module. Override this method to set up noise."""
        self._noise_module = p_err
    
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
        compiled_circuit = transpile(self.circuit, aer_sim, optimization_level=0)
        job = execute(compiled_circuit, aer_sim, shots=shots, noise_model=self._noise_module)
        job_monitor(job)
        counts = job.result().get_counts()
        self.plot_results(counts, f"{self._test_name} Results", f"{self._test_name}_histogram")
        # Check if error correction was successful
        self.check_ec(counts)
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

    def check_ec(self, counts):
        """Check if error correction was successful.
        
        Args:
            counts (dict): Dictionary of measurement results
            
        Returns:
            bool: True if error correction was successful, False otherwise
        """
        # Check if all measurements are 0
        all_zeros = all(int(key[-1]) == 0 for key in counts.keys())
        # If should_ec_fail is True, we expect failure (all_zeros should be False)
        # If should_ec_fail is False, we expect success (all_zeros should be True)
        success = all_zeros != self._should_ec_fail
        # Print colored result
        if success:
            print(f"{Fore.GREEN}Test Succeeded{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Test Failed{Style.RESET_ALL}")
            
        return success

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