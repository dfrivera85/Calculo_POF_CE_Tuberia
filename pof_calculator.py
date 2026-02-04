import numpy as np
from scipy import stats

def calculate_folias_factor(length, diameter, thickness):
    """
    Calculates the Folias Factor (M) for B31G Modified.
    M = sqrt(1 + 0.4 * L^2 / (D * t))
    """
    return np.sqrt(1 + 0.4 * (length**2) / (diameter * thickness))

def calculate_hoop_stress(pressure, diameter, thickness):
    """
    Calculates Hoop Stress (Barlow's formula approximation).
    Sigma_h = (P * D) / (2 * t)
    
    Args:
        pressure: Internal Pressure in PSI.
        diameter: Pipe Diameter in mm.
        thickness: Wall Thickness in mm.
        
    Returns:
        Hoop Stress in PSI.
    """
    return (pressure * diameter) / (2 * thickness)

def calculate_critical_depth(hoop_stress, flow_stress, thickness, folias_factor):
    """
    Calculates Critical Depth based on Modified B31G inversion.
    d_crit = (t / 0.85) * (Sigma_h - S_flow) / (Sigma_h/M - S_flow)
    
    Returns critical depth in mm. 
    Clamps result to thickness (cannot be physically deeper than wall for this check logic, though math might diverge).
    """
    # Avoid division by zero if (Sigma_h/M - S_flow) is 0
    denominator = (hoop_stress / folias_factor) - flow_stress
    
    # Handle singularities or invalid physics (e.g., Hoop > Flow, should be failure)
    # Ideally should be vectorized.
    
    # Formula rearrangement:
    # Failure if Sigma_h > S_flow_modified
    # d_crit calculation:
    
    numerator = hoop_stress - flow_stress
    
    # If denominator is 0 or positive (when Hoop/M >= Flow, meaning even with full wall it fails? No.)
    # B31G Logic: P_fail = ...
    # We solve for d/t.
    
    # Let's stick to the algebraic inversion:
    # d_crit = (t / 0.85) * numerator / denominator
    
    ##raw_d_crit = (thickness / 0.85) * (numerator / denominator)
    raw_d_crit = ((hoop_stress-flow_stress)*thickness)/(((hoop_stress/folias_factor)-flow_stress)*0.85)
    # If raw_d_crit is negative or > t, handle it.
    # If denominator is positive (unlikely for normal operation unless Hoop/M > Flow), it implies burst.
    # Standard usage: Denominator is usually negative (Hoop/M < Flow).
    # Numerator (Hoop - Flow) is usually negative.
    # So Negative / Negative = Positive.
    
    return np.clip(raw_d_crit, 0, thickness)

def calculate_pof_analytics(current_depths, current_lengths, pressure, diameter, thickness, smys, std_devs):
    """
    Calculates POF using analytical approximation (inverted B31G Mod).
    
    Args:
        current_depths (np.array): Mean depths (mm).
        current_lengths (np.array): Defect lengths (mm).
        pressure (float): Operating pressure (psi).
        diameter (float): Pipe outer diameter (mm).
        thickness (float): Wall thickness (mm).
        smys (float): Yield strength (psi).
        std_devs (np.array): Standard deviation of defect depth distribution (mm).
        
    Returns:
        np.array: Probability of Failure (0.0 to 1.0).
    """
    # 1. Properties
    flow_stress = smys + 10000 # B31G Modified standard
    hoop_stress = calculate_hoop_stress(pressure, diameter, thickness)
    
    # 2. Geometry Factors
    folias_factor = calculate_folias_factor(current_lengths, diameter, thickness)
    
    # 3. Limit State (Critical Depth)
    d_crit = calculate_critical_depth(hoop_stress, flow_stress, thickness, folias_factor)
    
    # 4. POF Calculation
    # Probability that Actual Depth > Critical Depth
    # Actual Depth ~ Normal(current_depths, std_devs)
    # POF = 1 - CDF(d_crit) = SF(d_crit)
    
    # Handle cases where std_dev is 0 to avoid nan (though unlikely in this model)
    # If std is 0, if depth > d_crit -> 1, else 0.
    
    # Using scipy.stats.norm.sf (Survival Function)
    # loc = mean, scale = std
    pof = stats.norm.sf(d_crit, loc=current_depths, scale=std_devs)
    
    return pof
