from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector
import os

# Get the backend simulator
aer_sim = Aer.get_backend('aer_simulator')
sv_sim = Aer.get_backend('statevector_simulator')

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
