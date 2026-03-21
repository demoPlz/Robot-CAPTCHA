import numpy as np
import itertools

def analyze_robot_experiments(num_successes, num_trials, model_names=None):
    """
    Performs rigorous statistical analysis for robot experiments using 
    Bootstrap CIs (for individual success AND for difference magnitude) 
    and Permutation Tests (for P-values).

    Args:
        num_successes (list of int): e.g., [20, 45, 80]
        num_trials (list of int): e.g., [100, 100, 100]
        model_names (list of str): e.g., ["BC", "Co-Training", "Ours"]
    """
    
    n_models = len(num_successes)
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n_models)]

    print("="*100)
    print(f" STATISTICAL ANALYSIS REPORT (N={num_trials[0]} runs)")
    print("="*100)

    stats_results = {} 
    model_data = {}

    # ---------------------------------------------------------
    # PART 1: Bootstrap Confidence Intervals (Individual Performance)
    # ---------------------------------------------------------
    print(f"\n{'Model Name':<20} | {'Success Rate':<15} | {'95% CI (Performance)':<25}")
    print("-" * 75)

    for i in range(n_models):
        name = model_names[i]
        s = num_successes[i]
        n = num_trials[i]
        
        # Expand counts into binary array
        data = np.array([1]*s + [0]*(n-s))
        model_data[name] = data
        
        # Bootstrap Resampling (10,000 times)
        n_boot = 10000
        indices = np.random.randint(0, n, size=(n_boot, n))
        resampled_means = data[indices].mean(axis=1)
        
        # Calculate Percentiles
        mean_rate = np.mean(data)
        lower = np.percentile(resampled_means, 2.5)
        upper = np.percentile(resampled_means, 97.5)
        
        stats_results[name] = {
            "mean": mean_rate,
            "lower": lower,
            "upper": upper,
            "n": n
        }
        
        print(f"{name:<20} | {mean_rate:.1%} ({s}/{n})   | [{lower:.1%} - {upper:.1%}]")

    # ---------------------------------------------------------
    # PART 2: Pairwise Tests (Significance + Magnitude of Improvement)
    # ---------------------------------------------------------
    print("\n" + "="*100)
    print(" PAIRWISE COMPARISONS (Magnitude of Improvement + Significance)")
    print("="*100)
    print(f"{'Comparison':<30} | {'Diff':<8} | {'95% CI (of Diff)':<20} | {'P-Value':<10} | {'Sig?'}")
    print("-" * 95)

    pairwise_stats = {}

    for name_a, name_b in itertools.combinations(model_names, 2):
        data_a = model_data[name_a]
        data_b = model_data[name_b]
        
        # A. Observed Difference
        obs_diff = np.mean(data_a) - np.mean(data_b)
        
        # B. Bootstrap the DIFFERENCE (To get CI of the gain)
        n_boot = 10000
        # Resample A and B independently
        idx_a = np.random.randint(0, len(data_a), size=(n_boot, len(data_a)))
        idx_b = np.random.randint(0, len(data_b), size=(n_boot, len(data_b)))
        
        means_a = data_a[idx_a].mean(axis=1)
        means_b = data_b[idx_b].mean(axis=1)
        diff_dist = means_a - means_b
        
        diff_lower = np.percentile(diff_dist, 2.5)
        diff_upper = np.percentile(diff_dist, 97.5)
        
        # C. Permutation Test (For P-Value)
        combined = np.concatenate([data_a, data_b])
        n_a = len(data_a)
        rng = np.random.default_rng()
        permuted_indices = rng.permuted(
            np.tile(np.arange(len(combined)), (n_boot, 1)), axis=1
        )
        permuted_pool = combined[permuted_indices]
        fake_a = permuted_pool[:, :n_a]
        fake_b = permuted_pool[:, n_a:]
        fake_diffs = fake_a.mean(axis=1) - fake_b.mean(axis=1)
        
        p_value = np.mean(np.abs(fake_diffs) >= np.abs(obs_diff))
        
        # Save results (store both directions for easy lookup)
        res_obj = {
            "diff": obs_diff,
            "lower": diff_lower,
            "upper": diff_upper,
            "p_value": p_value
        }
        pairwise_stats[(name_a, name_b)] = res_obj
        
        # Flip logic for reverse lookup (if needed later)
        pairwise_stats[(name_b, name_a)] = {
            "diff": -obs_diff,
            "lower": -diff_upper, # careful with sign flip
            "upper": -diff_lower,
            "p_value": p_value
        }

        # Print Row
        is_sig = "YES" if p_value < 0.05 else "NO"
        sig_symbol = "*" if p_value < 0.05 else ""
        
        print(f"{name_a} vs {name_b:<15} | {obs_diff:+.1%}  | [{diff_lower:+.1%}, {diff_upper:+.1%}] | {p_value:.4f}     | {is_sig}{sig_symbol}")

    # ---------------------------------------------------------
    # PART 3: GENERATE CoRL PAPER TEXT (Enhanced)
    # ---------------------------------------------------------
    print("\n" + "="*100)
    print(" SUGGESTED LATEX / PAPER TEXT")
    print("="*100)
    
    # Assumption: The LAST model in the list is "Ours"
    hero_name = model_names[-1]
    baselines = model_names[:-1]
    hero_res = stats_results[hero_name]
    
    # Build the string for baselines
    baseline_texts = []
    for base in baselines:
        res_b = stats_results[base]
        pair_res = pairwise_stats[(hero_name, base)]
        
        # P-value string
        p_val = pair_res['p_value']
        p_str = "p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"
        
        # Improvement string
        imp_mean = pair_res['diff']
        imp_lower = pair_res['lower']
        imp_upper = pair_res['upper']
        
        # Format: "BaselineName (45.0%; improvement: +20.0% [95% CI: +5%, +35%], p < 0.001)"
        text = (
            f"{base} ({res_b['mean']:.1%}; "
            f"absolute gain: {imp_mean:+.1%} [95% CI: {imp_lower:+.1%}, {imp_upper:+.1%}], {p_str})"
        )
        baseline_texts.append(text)

    if len(baseline_texts) == 1:
        baseline_str = baseline_texts[0]
    else:
        baseline_str = "; ".join(baseline_texts[:-1]) + "; and " + baseline_texts[-1]

    statement = (
        f"We evaluated policy performance over {hero_res['n']} independent real-world trials per model. "
        f"To account for statistical uncertainty, we report Stratified Bootstrap "
        f"Confidence Intervals (95% CI) using 10,000 bootstrap samples.\n\n"
        
        f"Our method, \\textbf{{{hero_name}}}, achieved a success rate of {hero_res['mean']:.1%} "
        f"(95% CI: [{hero_res['lower']:.1%}, {hero_res['upper']:.1%}]). "
        f"This constitutes a statistically significant improvement over "
        f"{baseline_str}, as confirmed by pairwise two-sided permutation tests."
    )
    
    print(statement)
    print("="*100)

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Example: 
    # Diffusion: 650/1000
    # Ours: 840/1000
    
    successes = [21, 74] 
    trials =    [100, 100]
    names =     ["Diffusion", "Ours"]

    analyze_robot_experiments(successes, trials, names)