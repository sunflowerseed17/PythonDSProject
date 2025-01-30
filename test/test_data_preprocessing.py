###############################################################################
#  IMPORTS
###############################################################################

import unittest
import os
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from data_preprocessing.data_preprocessing import preprocess_text, create_folder

###############################################################################
# DATA PREPROCESSING TEST
###############################################################################
# The outputs are removed after being created, but by removing the tearDown() func, the ouputs save

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        # Set up paths
        self.raw_input_folder = "test/test_data/raw"
        self.output_folder = "test/test_data/preprocessed"
        create_folder(self.output_folder)

    def test_preprocess_and_save(self):
        # Loop through each category and preprocess/save the post
        for category in ["breastcancer", "standard", "depression"]:
            input_folder = os.path.join(self.raw_input_folder, category)
            self.assertTrue(os.path.exists(input_folder), f"Input folder {input_folder} does not exist.")

            # Get the post file from the input folder
            input_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
            self.assertTrue(len(input_files) > 0, f"No .txt files found in {input_folder}.")
            input_file_path = os.path.join(input_folder, input_files[0])

            # Read the post content
            with open(input_file_path, "r", encoding="utf-8") as file:
                post_content = file.read()

            # Preprocess the post content
            processed_text = preprocess_text(post_content)

            # Define the output file path and ensure the directory exists
            output_folder = os.path.join(self.output_folder, category)
            os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
            output_file = os.path.join(output_folder, f"{category}_preprocessed.txt")
            
            try:
                # Save the preprocessed content to the output folder
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(processed_text)
            except Exception as e:
                self.fail(f"Failed to save preprocessed file for category {category}: {e}")

            # Verify that the file was saved and contains the processed text
            self.assertTrue(os.path.exists(output_file), f"Output file {output_file} not found.")
            with open(output_file, "r", encoding="utf-8") as file:
                content = file.read()
                self.assertIn(processed_text, content, "Processed text not found in output file.")

    #def tearDown(self):
    #    if os.path.exists(self.output_folder):
    #        for file in os.listdir(self.output_folder):
    #            os.remove(os.path.join(self.output_folder, file))

if __name__ == "__main__":
    unittest.main()
