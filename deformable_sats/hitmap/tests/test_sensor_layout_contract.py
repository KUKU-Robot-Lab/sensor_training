import unittest

import torch

from training.data.sensor_layout import build_sensor_positions


class SensorLayoutContractTest(unittest.TestCase):
    def test_build_sensor_positions_matches_documented_x_descending_layout(self) -> None:
        positions = build_sensor_positions()

        self.assertEqual(tuple(positions.shape), (16, 2))
        expected_first_row = torch.tensor(
            [
                [9.75, -9.75],
                [3.25, -9.75],
                [-3.25, -9.75],
                [-9.75, -9.75],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(positions[:4], expected_first_row)

    def test_build_sensor_positions_increases_y_by_row(self) -> None:
        positions = build_sensor_positions()

        self.assertEqual(positions[0, 1].item(), -9.75)
        self.assertEqual(positions[4, 1].item(), -3.25)
        self.assertEqual(positions[8, 1].item(), 3.25)
        self.assertEqual(positions[12, 1].item(), 9.75)


if __name__ == "__main__":
    unittest.main()
