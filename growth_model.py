import numpy as np

def grow_defects(current_depths, current_lengths, corrosion_rates, max_depths, time_step=1.0):
    """
    Applies corrosion growth to defects over time using vectorization.

    Args:
        current_depths (np.array): Depths at t-1 (mm).
        current_lengths (np.array): Lengths at t-1 (mm).
        corrosion_rates (np.array): Corrosion rates (mm/year).
        max_depths (np.array): Wall thickness (mm), limit for depth.
        time_step (float): Time increment in years.

    Returns:
        tuple: (next_depths, next_lengths) as np.arrays.
    """
    # Calculate next state
    growth_amount = corrosion_rates * time_step
    
    # Apply growth
    raw_next_depths = current_depths + growth_amount
    next_lengths = current_lengths + growth_amount
    
    # Enforce physical limits (depth <= wall thickness)
    next_depths = np.minimum(raw_next_depths, max_depths)
    
    return next_depths, next_lengths

