# Shor Code Error Correction Tests

This project contains a comprehensive test suite for the 9-qubit Shor code, a quantum error correction code that can protect quantum information from both bit flip (X) and phase flip (Z) errors.

## Structure

The project is organized into the following files:

- `utils.py`: Common utilities and functions used across all tests
- `test1_bit_flip.py`: Test for bit flip (X) error correction
- `test2_phase_flip.py`: Test for phase flip (Z) error correction
- `test3_y_error.py`: Test for simultaneous bit and phase flip (Y) error correction
- `test4_multiple_errors.py`: Test for multiple errors beyond the code's capability
- `test5_random_errors.py`: Statistical analysis with random error injection
- `test6_noise_model.py`: Tests with IBM quantum hardware noise model
- `test7_targeted_errors.py`: Tests with targeted error correction
- `main.py`: Script to run all tests and generate a complete report

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Tests

You can run individual tests:

```bash
python test1_bit_flip.py
python test2_phase_flip.py
# ... etc.
```

Or run all tests together:

```bash
python main.py
```

## Output

The tests will generate:
- Detailed text output describing the results of each test
- Circuit diagrams saved as PNG files in the `images/` directory
- Performance plots and statistics for each test

## Test Descriptions

1. **Bit Flip Test**: Tests the code's ability to correct X errors
2. **Phase Flip Test**: Tests the code's ability to correct Z errors
3. **Y Error Test**: Tests correction of simultaneous X and Z errors
4. **Multiple Errors Test**: Demonstrates the code's limitations
5. **Random Errors Test**: Statistical analysis of error correction
6. **Noise Model Test**: Tests under realistic quantum hardware noise
7. **Targeted Errors Test**: Tests with controlled error injection

## Results

Results are displayed in both text format and visualizations. The test suite generates:
- Success rates for different types of errors
- Fidelity measurements
- Comparison plots
- Circuit diagrams
- Statistical analyses

All images are saved in the `images/` directory for further analysis. 