
import unittest
import numpy as np
from scipy import stats
from pof_calculator import (
    calculate_folias_factor,
    calculate_hoop_stress,
    calculate_critical_depth,
    calculate_pof_analytics
)

class TestPOFCalculator(unittest.TestCase):

    def test_folias_factor(self):
        """Test Folias Factor (M) calculation."""
        # Case 1: Length = 0 -> M = 1
        L = 0
        D = 500
        t = 10
        M = calculate_folias_factor(np.array([L]), D, t)
        self.assertAlmostEqual(M[0], 1.0)

        # Case 2: Known value check
        # L=100, D=500, t=10 => L^2/(Dt) = 10000/5000 = 2
        # M = sqrt(1 + 0.4 * 2) = sqrt(1.8) approx 1.3416
        L = 100
        M = calculate_folias_factor(np.array([L]), D, t)
        self.assertAlmostEqual(M[0], np.sqrt(1.8))

    def test_hoop_stress(self):
        """Test Hoop Stress calculation."""
        # P = 10 MPa, D = 500 mm, t = 10 mm
        # Hoop = (10 * 500) / (2 * 10) = 5000 / 20 = 250 MPa
        P = 10
        D = 500
        t = 10
        stress = calculate_hoop_stress(P, D, t)
        self.assertEqual(stress, 250.0)

    def test_critical_depth(self):
        """Test Critical Depth calculation (B31G Modified Inversion)."""
        # Test case: Failure expected immediately if Hoop >= Flow Stress (Should not happen in operation usually, but conceptually)
        # If Hoop Stress = Flow Stress, denominator becomes 0 or undefined conceptually if M=1, but let's check basic logic.
        
        # Standard case
        # t = 10 mm
        # Flow Stress = 400 MPa
        # Hoop Stress = 200 MPa
        # M = 2.0
        # d_crit = (10 / 0.85) * (200 - 400) / (200/2 - 400)
        #        = 11.764 * (-200) / (100 - 400)
        #        = 11.764 * (-200) / (-300)
        #        = 11.764 * 0.666 = 7.84 mm approx
        
        t = 10.0
        flow_stress = 400.0
        hoop_stress = 200.0
        folias_factor = np.array([2.0])
        
        d_crit = calculate_critical_depth(hoop_stress, flow_stress, t, folias_factor)
        self.assertAlmostEqual(d_crit[0], 7.843, places=2)

    def test_pof_calculation(self):
        """Test POF probability calculation."""
        # Setup:
        # Critical Depth = 5.0 mm
        # Defect Depth Mean = 5.0 mm
        # Defect Std Dev = 1.0 mm
        # Expected POF = 0.5 (Normal distribution centered on limit)
        
        # We need to reverse engineer inputs to get d_crit = 5.0
        # Let's mock calculate_critical_depth inside the test or just trust the integration.
        # Actually, let's just pass values that result in known critical depths.
        
        # Simply testing the analytical integration if we could inject d_crit, but the function computes it internally.
        # So we construct a case.
        
        diameter = 500
        thickness = 10
        pressure = 10 # Hoop = 250
        smys = 300 # Flow = 369
        # We need M such that d_crit matches something.
        # Let's just run it and check directionality.
        
        current_depths = np.array([2.0, 9.5]) # One small, one definitely large (critical ~ 7.66)
        current_lengths = np.array([100.0, 100.0]) # Fixed L
        std_devs = np.array([0.5, 0.5]) # Tight distribution
        
        # Expect POF low for depth 2.0, POF high for depth 9.5
        
        pofs = calculate_pof_analytics(
            current_depths, current_lengths, pressure, diameter, thickness, smys, std_devs
        )
        
        self.assertTrue(pofs[0] < 0.1, f"Expected safe POF < 0.1, got {pofs[0]}") 
        self.assertTrue(pofs[1] > 0.9, f"Expected failure POF > 0.9, got {pofs[1]}") # 9.5 vs 7.66 with std 0.5 is > 3 sigma failure
        self.assertTrue(all(0 <= p <= 1 for p in pofs))

if __name__ == '__main__':
    unittest.main()
