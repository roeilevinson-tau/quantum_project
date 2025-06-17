import test1_bit_flip
import test2_phase_flip
import test3_y_error
# import test4_multiple_errors
# import test5_noise_module_errors


def main():
    """Run all tests and print results"""
    print("Running all Shor code tests...\n")
    
    # Run tests
    test1_results = test1_bit_flip.run_test()
    test2_results = test2_phase_flip.run_test()
    test3_results = test3_y_error.run_test()
    # test4_results = test4_multiple_errors.run_test()
    # test5_results = test5_noise_module_errors.run_test()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main() 