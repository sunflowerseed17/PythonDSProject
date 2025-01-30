import unittest

if __name__ == "__main__":
    # Discover and run all tests in the 'test' directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir="test", pattern="*.py")

    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)