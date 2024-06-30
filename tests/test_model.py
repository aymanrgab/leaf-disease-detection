import unittest
import torch
import sys
import os

# Add parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

from src.model import get_model
from src.data_preprocessing import PlantVillageDataset

# Determine the number of classes from the PlantVillageDataset
num_classes = len(PlantVillageDataset('with-augmentation').classes)

class TestModel(unittest.TestCase):
    """
    Unit test class for testing the model's functionality.
    """

    def test_model_output_shape(self):
        """
        Test to ensure the model's output shape is as expected.
        
        This test checks if the model's output shape matches the expected shape
        given an input tensor of shape (1, 3, 224, 224).
        """
        model = get_model(num_classes)
        input_tensor = torch.randn(1, 3, 224, 224)
        output = model(input_tensor)
        self.assertEqual(output.shape, torch.Size([1, 38]))

    def test_model_forward_pass(self):
        """
        Test to ensure the model's forward pass works without raising exceptions.
        
        This test checks if the model can process an input tensor of shape (1, 3, 224, 224)
        without raising any exceptions.
        """
        model = get_model(num_classes)
        input_tensor = torch.randn(1, 3, 224, 224)
        try:
            output = model(input_tensor)
        except Exception as e:
            self.fail(f"Model forward pass raised an exception: {str(e)}")

    def test_model_parameters(self):
        """
        Test to ensure the model has trainable parameters.
        
        This test checks if the model has any trainable parameters by verifying
        that the length of the parameters list is greater than zero.
        """
        model = get_model(num_classes)
        params = list(model.parameters())
        self.assertGreater(len(params), 0)

if __name__ == '__main__':
    unittest.main()