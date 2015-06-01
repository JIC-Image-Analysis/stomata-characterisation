import unittest
import os
import os.path
import shutil

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, '..', 'data', 'raw')
SCRIPTS_DIR = os.path.join(HERE, '..', 'scripts')
INPUT_FILE = os.path.join(DATA_DIR, '2014-03-20-Lti6b-GFP-ABA-time-series.lif')
TMP_DIR = os.path.join(HERE, 'tmp')

class UnitTests(unittest.TestCase):
    
    def test_unpack_data(self):
        from jicimagelib.image import ImageCollection
        from scripts.util import unpack_data
        image_collection = unpack_data(INPUT_FILE)
        self.assertTrue(isinstance(image_collection, ImageCollection))

class FunctionalTests(unittest.TestCase):

    def setUp(self):
        if not os.path.isdir(TMP_DIR):
            os.mkdir(TMP_DIR)

    def tearDown(self):
        shutil.rmtree(TMP_DIR)

    def test_find_stomata(self):
        script = os.path.join(SCRIPTS_DIR, 'find_stomata.py')
        cmd = 'python {} {} 8 9 --output_dir {}'.format(
            script, INPUT_FILE, TMP_DIR)
        os.system(cmd)
        for fname in ["1_max_intensity_projection.png",
            "2_threshold_otsu.png",
            "3_find_connected_components.png",
            "4_max_intensity_projection.png",
            "annotated_image.png",
            "stomata_border.png",
            "stomata.png"]:
            fpath = os.path.join(TMP_DIR, fname)
            self.assertTrue(os.path.isfile(fpath))

    def test_calculate_opening(self):
        from scripts.util import unpack_data, stomata_lookup
        from scripts.calculate_opening import calculate_opening
        
        region_id, series_ids = stomata_lookup(0)
        image_collection = unpack_data(INPUT_FILE)
        distance = calculate_opening(image_collection, 0, series_ids[0])
        self.assertEqual(round(distance, 2), 1.24)
        
if __name__ == "__main__":
    unittest.main()
