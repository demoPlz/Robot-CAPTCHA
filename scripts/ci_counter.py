import numpy as np

def calculate_single_ci(x, y, n_boot=10000):
    """
    Calculates the 95% Bootstrap Confidence Interval for a proportion (x/y).
    
    Args:
        x (int): Number of successes.
        y (int): Total number of trials.
        n_boot (int): Number of bootstrap iterations (default: 10,000).
        
    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    if y <= 0 or x > y or x < 0:
        raise ValueError("Invalid input: x must be between 0 and y, and y > 0.")

    # 1. Expand counts into a binary array (1 for success, 0 for failure)
    data = np.array([1] * x + [0] * (y - x))
    
    # 2. Bootstrap Resampling
    # Generate random indices to sample with replacement
    indices = np.random.randint(0, y, size=(n_boot, y))
    resampled_means = data[indices].mean(axis=1)
    
    # 3. Calculate Mean and 95% Percentiles
    mean_rate = np.mean(data)
    lower = np.percentile(resampled_means, 2.5)
    upper = np.percentile(resampled_means, 97.5)
    
    # 4. Output Results
    print("="*50)
    print(f" STATISTIC: {x}/{y} ({mean_rate:.1%})")
    print("="*50)
    print(f" 95% CI: [{lower:.1%}, {upper:.1%}]")
    print("="*50)
    
    return mean_rate, lower, upper

if __name__ == "__main__":
    # Example Usage: 74 successes out of 100 trials
    successes = 30
    trials = 100
    
    calculate_single_ci(successes, trials)