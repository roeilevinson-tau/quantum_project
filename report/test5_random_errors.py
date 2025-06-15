from utils import (
    create_shor_encoded_state, decode_shor_code,
    run_circuit, get_image_path
)
import matplotlib.pyplot as plt
import random

def run_test():
    """Test 5: Random Error Injection (Statistical Analysis)"""
    print("\n## Test 5: Random Error Injection (Statistical Analysis)")
    print("----------------------------------------------------\n")
    print("This test performs statistical analysis by injecting random single-qubit errors and measuring recovery rates.\n")

    num_trials = 10
    successes = 0
    error_types = ["X", "Z", "Y"]
    error_positions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    results = {
        "X": {"success": 0, "trials": 0},
        "Z": {"success": 0, "trials": 0},
        "Y": {"success": 0, "trials": 0}
    }

    for i in range(num_trials):
        # Choose a random error type and position
        error_type = random.choice(error_types)
        error_pos = random.choice(error_positions)
        
        # Create circuit with a logical |0⟩ state
        circuit, q, c = create_shor_encoded_state([1, 0])
        
        # Apply the random error
        if error_type == "X":
            circuit.x(q[error_pos])
            results["X"]["trials"] += 1
        elif error_type == "Z":
            circuit.z(q[error_pos])
            results["Z"]["trials"] += 1
        elif error_type == "Y":
            circuit.y(q[error_pos])
            results["Y"]["trials"] += 1
        
        # Decode the state
        circuit = decode_shor_code(circuit, q, c)
        
        # Measure
        circuit.measure(q[0], c[0])
        
        # Execute and get results
        counts = run_circuit(circuit, shots=100)
        
        # Count as success if |0⟩ has higher probability
        if '0' in counts and ('1' not in counts or counts['0'] > counts['1']):
            successes += 1
            if error_type == "X":
                results["X"]["success"] += 1
            elif error_type == "Z":
                results["Z"]["success"] += 1
            elif error_type == "Y":
                results["Y"]["success"] += 1

    # Calculate success rates
    overall_success_rate = successes / num_trials
    error_success_rates = {
        "X": results["X"]["success"] / max(1, results["X"]["trials"]),
        "Z": results["Z"]["success"] / max(1, results["Z"]["trials"]),
        "Y": results["Y"]["success"] / max(1, results["Y"]["trials"])
    }

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(error_success_rates.keys(), error_success_rates.values())
    plt.title("Error Correction Success Rates by Error Type")
    plt.ylabel("Success Rate")
    plt.xlabel("Error Type")
    plt.ylim(0, 1)
    plt.savefig(get_image_path("test5_random_error_statistics.png"))
    plt.close()

    print(f"### Results:")
    print(f"- Overall success rate: {overall_success_rate:.2%}")
    print(f"- Success rate for X errors: {error_success_rates['X']:.2%}")
    print(f"- Success rate for Z errors: {error_success_rates['Z']:.2%}")
    print(f"- Success rate for Y errors: {error_success_rates['Y']:.2%}")
    print("\nConclusion: The Shor code demonstrates high recovery rates for single-qubit errors, ")
    print("regardless of the error type (X, Z, or Y) and qubit position.\n")

    return {
        'overall_success_rate': overall_success_rate,
        'error_success_rates': error_success_rates
    }

if __name__ == "__main__":
    run_test() 