import unittest
import sys
import os

# Add parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import PlantVillageDataset

class TestDataPreprocessing(unittest.TestCase):
    def test_plant_village_dataset(self):
        dataset = PlantVillageDataset(subset='without-augmentation')
        self.assertGreater(len(dataset), 0)

    def test_plant_village_dataset_with_augmentation(self):
        dataset = PlantVillageDataset(subset='with-augmentation')
        self.assertGreater(len(dataset), 0)

if __name__ == '__main__':
    unittest.main()
