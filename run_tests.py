import unittest
import logging

###############################################################################
# LOGGER CONFIGURATION
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

###############################################################################
# CONFIGURATION CONSTANTS
###############################################################################
TEST_DIRECTORY = "test"
TEST_PATTERN = "*.py"
TEST_VERBOSITY = 2

if __name__ == "__main__":
    logger.info("Discovering tests in directory '%s' with pattern '%s'...", TEST_DIRECTORY, TEST_PATTERN)
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=TEST_DIRECTORY, pattern=TEST_PATTERN)

    logger.info("Running tests with verbosity %d...", TEST_VERBOSITY)
    test_runner = unittest.TextTestRunner(verbosity=TEST_VERBOSITY)
    test_runner.run(test_suite)