
import unittest
import numpy as np
import pandas as pd
from growth_model import grow_defects

class TestGrowthModel(unittest.TestCase):

    def test_basic_growth(self):
        """Test linear growth calculation."""
        current_depths = np.array([1.0, 2.0, 3.0])
        current_lengths = np.array([10.0, 20.0, 30.0])
        corrosion_rates = np.array([0.1, 0.2, 0.3])
        max_depths = np.array([10.0, 10.0, 10.0]) # Plenty of room
        time_step = 1.0

        next_depths, next_lengths = grow_defects(
            current_depths, current_lengths, corrosion_rates, max_depths, time_step
        )

        np.testing.assert_array_almost_equal(next_depths, np.array([1.1, 2.2, 3.3]))
        np.testing.assert_array_almost_equal(next_lengths, np.array([10.1, 20.2, 30.3]))

    def test_growth_limited_by_wall_thickness(self):
        """Test that depth does not exceed wall thickness."""
        current_depths = np.array([9.5, 5.0])
        current_lengths = np.array([10.0, 10.0])
        corrosion_rates = np.array([1.0, 1.0])
        max_depths = np.array([10.0, 10.0]) # Wall thickness
        time_step = 1.0

        next_depths, next_lengths = grow_defects(
            current_depths, current_lengths, corrosion_rates, max_depths, time_step
        )

        # First one should be capped at 10.0, second one grows to 6.0
        np.testing.assert_array_almost_equal(next_depths, np.array([10.0, 6.0]))
        # Lengths are not capped by wall thickness
        np.testing.assert_array_almost_equal(next_lengths, np.array([11.0, 11.0]))

    def test_zero_growth(self):
        """Test no change when corrosion rate is zero."""
        current_depths = np.array([5.0])
        current_lengths = np.array([10.0])
        corrosion_rates = np.array([0.0])
        max_depths = np.array([10.0])
        time_step = 5.0

        next_depths, next_lengths = grow_defects(
            current_depths, current_lengths, corrosion_rates, max_depths, time_step
        )

        np.testing.assert_array_almost_equal(next_depths, np.array([5.0]))
        np.testing.assert_array_almost_equal(next_lengths, np.array([10.0]))

    def test_custom_time_step(self):
        """Test growth with non-default time step."""
        current_depths = np.array([1.0])
        current_lengths = np.array([10.0])
        corrosion_rates = np.array([0.1])
        max_depths = np.array([10.0])
        time_step = 0.5 # Half a year

        next_depths, next_lengths = grow_defects(
            current_depths, current_lengths, corrosion_rates, max_depths, time_step
        )

        np.testing.assert_array_almost_equal(next_depths, np.array([1.05]))
        np.testing.assert_array_almost_equal(next_lengths, np.array([10.05]))

if __name__ == '__main__':
    unittest.main()
