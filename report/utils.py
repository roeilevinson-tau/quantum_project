from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import state_fidelity, Statevector
import os

# Get the backend simulator
aer_sim = Aer.get_backend('aer_simulator')
sv_sim = Aer.get_backend('statevector_simulator')

# Create directory for images if it doesn't exist
def ensure_images_dir():
    """Create the images directory if it doesn't exist."""
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir

# Function to get the full path for an image file
def get_image_path(filename):
    """Get the full path for an image file."""
    return os.path.join(ensure_images_dir(), filename)

# Define a function to create the Shor encoded state
def create_shor_encoded_state(initial_state=[1, 0]):
    """
    Create a Shor code encoded state
    initial_state: list of 2 complex numbers for the |0⟩ and |1⟩ amplitudes
    """
    # Normalize the initial state (just in case)
    norm = np.sqrt(abs(initial_state[0])**2 + abs(initial_state[1])**2)
    initial_state = [initial_state[0]/norm, initial_state[1]/norm]
    
    q = QuantumRegister(9, 'q')
    c = ClassicalRegister(9, 'c')
    circuit = QuantumCircuit(q, c)
    
    # Initialize the first qubit to our chosen state
    if initial_state[1] != 0:
        # We need to prepare arbitrary state
        circuit.initialize(initial_state, q[0])
    
    # Encoding
    circuit.cx(q[0], q[3])
    circuit.cx(q[0], q[6])
    
    circuit.h(q[0])
    circuit.h(q[3])
    circuit.h(q[6])
    
    circuit.cx(q[0], q[1])
    circuit.cx(q[3], q[4])
    circuit.cx(q[6], q[7])
    
    circuit.cx(q[0], q[2])
    circuit.cx(q[3], q[5])
    circuit.cx(q[6], q[8])
    
    return circuit, q, c

# Define a function to decode the Shor code
def decode_shor_code(circuit, q, c):
    """Add decoding operations to the circuit"""
    # First level of error correction - correct bit flips within each block
    # Block 1
    circuit.cx(q[0], q[1])  # Compare q[0] and q[1]
    circuit.cx(q[0], q[2])  # Compare q[0] and q[2]
    circuit.ccx(q[1], q[2], q[0])  # Majority vote
    
    # Block 2
    circuit.cx(q[3], q[4])
    circuit.cx(q[3], q[5])
    circuit.ccx(q[4], q[5], q[3])
    
    # Block 3
    circuit.cx(q[6], q[7])
    circuit.cx(q[6], q[8])
    circuit.ccx(q[7], q[8], q[6])
    
    # Convert to phase basis for phase error correction
    circuit.h(q[0])
    circuit.h(q[3])
    circuit.h(q[6])
    
    # Second level - correct phase flips
    circuit.cx(q[0], q[3])  # Compare blocks
    circuit.cx(q[0], q[6])
    circuit.ccx(q[3], q[6], q[0])  # Majority vote
    
    # Convert back to computational basis
    circuit.h(q[0])
    
    return circuit

# Function to run a circuit and get counts
def run_circuit(circuit, shots=1000):
    job = execute(circuit, aer_sim, shots=shots)
    job_monitor(job)
    return job.result().get_counts()

# Function to run a circuit and get the statevector
def get_statevector(circuit):
    job = execute(circuit, sv_sim)
    return job.result().get_statevector()

# Function to visualize results
def plot_results(results_dict, title, filename=None):
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