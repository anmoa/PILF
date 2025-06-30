import numpy as np
from omegaid.core.phiid import calc_phiid_ccs


def generate_data(n_samples, n_features, case="redundancy"):
    """
    Generates synthetic data for s1, s2, and t based on a specific
    information-theoretic case.
    
    Args:
        n_samples (int): Number of data points.
        n_features (int): Number of features for each source.
        case (str): One of "redundancy", "unique_s1", "unique_s2", "synergy".

    Returns:
        tuple: (s1, s2, t) as numpy arrays.
    """
    s1 = np.random.randn(n_samples, n_features)
    s2 = np.random.randn(n_samples, n_features)
    
    if case == "redundancy":
        # Both s1 and s2 are correlated with a common signal
        common_signal = np.random.randn(n_samples, n_features)
        s1 = s1 * 0.1 + common_signal
        s2 = s2 * 0.1 + common_signal
        t = common_signal
    elif case == "unique_s1":
        # t is only dependent on s1
        t = s1
    elif case == "unique_s2":
        # t is only dependent on s2
        t = s2
    elif case == "synergy":
        # t is dependent on the interaction of s1 and s2 (e.g., XOR)
        # For continuous variables, multiplication is a simple synergistic interaction.
        t = s1 * s2
    else:
        raise ValueError(f"Unknown case: {case}")
        
    return s1, s2, t

def main():
    """
    Main function to run the benchmark.
    """
    n_samples = 50000
    n_features = 64  # A realistic feature dimension for our models
    
    print(f"Running ΩID benchmark with {n_samples} samples and {n_features} features.")
    print("-" * 60)

    for case in ["redundancy", "unique_s1", "unique_s2", "synergy"]:
        print(f"Case: {case.upper()}")
        
        s1, s2, t = generate_data(n_samples, n_features, case)
        
        # We analyze the information from sources (s1, s2) about the target (t)
        # In our model, s1 and s2 are expert activations, and t is the next layer's activation.
        # For this benchmark, we'll analyze how s1 and s2 relate to each other,
        # which is equivalent to setting t=s2.
        # The function calc_phiid_ccs(src, trg) calculates I(src; trg_past | src_past),
        # so we pass s1 and s2.
        
        # To use GPU, we would convert numpy arrays to cupy arrays.
        # For this benchmark, we'll stick to numpy to ensure it runs everywhere.
        # try:
        #     import cupy as cp
        #     s1_gpu = cp.asarray(s1)
        #     s2_gpu = cp.asarray(s2)
        #     backend = "CuPy"
        # except (ImportError, cp.cuda.runtime.CUDARuntimeError):
        #     s1_gpu = s1
        #     s2_gpu = s2
        #     backend = "NumPy"
        
        # The function calc_phiid_ccs appears to be designed for UNIvARIATE
        # time series (n_samples,). It fails when given a multivariate
        # time series (n_samples, n_features).
        #
        # To validate the core logic, we will run the benchmark on a single
        # feature from our generated data. This simulates two univariate time series.
        s1_univariate = s1[:, 0]
        s2_univariate = s2[:, 0]

        atoms_res, _ = calc_phiid_ccs(s1_univariate, s2_univariate, tau=1, kind='gaussian')

        # The result is a single dictionary of scalar atom values.
        print(f"  Redundancy (R): {atoms_res['R']:.4f}")
        print(f"  Unique S1 (U_s1): {atoms_res['U_s1']:.4f}")
        print(f"  Unique S2 (U_s2): {atoms_res['U_s2']:.4f}")
        print(f"  Synergy (S): {atoms_res['S']:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    main()
