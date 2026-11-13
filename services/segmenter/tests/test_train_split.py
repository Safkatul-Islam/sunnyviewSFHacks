import os
import sys
import unittest

_TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train_roofsat"))
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

from dataset import RoofSatSample, split_train_val, split_train_val_test  # noqa: E402


def _samples(n: int) -> list:
    return [RoofSatSample(img_path=f"img/{i:04d}.png", mask_path=f"mask/{i:04d}.png") for i in range(n)]


def _paths(samples: list) -> set:
    return {s.img_path for s in samples}


class TestSplitTrainValTest(unittest.TestCase):
    def test_splits_are_disjoint_and_cover_every_sample(self):
        samples = _samples(550)
        train, val, test = split_train_val_test(samples, val_frac=0.15, test_frac=0.15)
        self.assertEqual((len(train), len(val), len(test)), (398, 70, 82))
        self.assertFalse(_paths(test) & (_paths(train) | _paths(val)))
        self.assertFalse(_paths(train) & _paths(val))
        self.assertEqual(_paths(train) | _paths(val) | _paths(test), _paths(samples))

    def test_test_set_is_fixed_across_seed_and_val_frac(self):
        samples = _samples(200)
        _, _, a = split_train_val_test(samples, val_frac=0.15, test_frac=0.2, seed=1)
        _, _, b = split_train_val_test(samples, val_frac=0.3, test_frac=0.2, seed=99)
        self.assertEqual(_paths(a), _paths(b))

    def test_test_set_ignores_input_order(self):
        samples = _samples(120)
        _, _, a = split_train_val_test(samples, val_frac=0.15, test_frac=0.25)
        _, _, b = split_train_val_test(list(reversed(samples)), val_frac=0.15, test_frac=0.25)
        self.assertEqual(_paths(a), _paths(b))

    def test_test_seed_changes_the_test_set(self):
        samples = _samples(200)
        _, _, a = split_train_val_test(samples, val_frac=0.15, test_frac=0.2, test_seed=1)
        _, _, b = split_train_val_test(samples, val_frac=0.15, test_frac=0.2, test_seed=2)
        self.assertNotEqual(_paths(a), _paths(b))

    def test_zero_test_frac_matches_the_original_split(self):
        samples = _samples(80)
        train, val, test = split_train_val_test(samples, val_frac=0.15, test_frac=0.0, seed=1337)
        old_train, old_val = split_train_val(samples, 0.15, seed=1337)
        self.assertEqual(test, [])
        self.assertEqual((train, val), (old_train, old_val))

    def test_rejects_invalid_test_frac(self):
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                split_train_val_test(_samples(50), val_frac=0.15, test_frac=bad)

    def test_rejects_a_test_frac_that_leaves_no_training_data(self):
        with self.assertRaises(ValueError):
            split_train_val_test(_samples(3), val_frac=0.5, test_frac=0.9)


if __name__ == "__main__":
    unittest.main()
