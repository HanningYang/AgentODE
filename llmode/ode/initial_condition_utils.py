"""Utilities for loading initial condition configurations and generating simulated data."""

import json
import numpy as np
from typing import Dict, Tuple, Optional
import os
from scipy.stats import multivariate_normal, norm


def load_ic_config(problem_name: str, config_dir: str = 'initial_conditions') -> dict:
    """Load initial condition configuration for a given problem.

    Args:
        problem_name: Name of the problem (e.g., 'aki', 'vasculitis')
        config_dir: Directory containing configuration files

    Returns:
        Dictionary containing the full configuration

    Raises:
        FileNotFoundError: If configuration file doesn't exist
    """
    config_path = os.path.join(config_dir, f'{problem_name}_config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Available configs: {list_available_configs(config_dir)}"
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def list_available_configs(config_dir: str = 'initial_conditions') -> list:
    """List all available initial condition configurations.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        List of problem names (without _config.json suffix)
    """
    if not os.path.exists(config_dir):
        return []

    configs = []
    for filename in os.listdir(config_dir):
        if filename.endswith('_config.json'):
            problem_name = filename.replace('_config.json', '')
            configs.append(problem_name)

    return configs


def calculate_lognormal_params(mean: float, std: float) -> Tuple[float, float]:
    """
    Calculate log-normal parameters μ and σ from linear-space mean and std.

    For X ~ LogNormal(μ, σ), we have:
        E[X] = exp(μ + σ²/2)
        Var[X] = [exp(σ²) - 1] * exp(2μ + σ²)

    Given mean and std in linear space, we solve for μ and σ:
        σ² = ln(1 + (std/mean)²)
        μ = ln(mean) - σ²/2

    Args:
        mean: Mean in linear space
        std: Standard deviation in linear space

    Returns:
        (mu, sigma): Parameters for log-normal distribution
    """
    variance = std ** 2
    sigma_squared = np.log(1 + variance / (mean ** 2))
    sigma = np.sqrt(sigma_squared)
    mu = np.log(mean) - sigma_squared / 2
    return mu, sigma


def approximate_log_space_correlation(
    linear_corr: np.ndarray,
    biomarker_types: list
) -> np.ndarray:
    """
    Approximate correlation matrix in transformed (log) space.

    For log-normal variables, the correlation in log-space is typically
    slightly higher than in linear space. This function provides a heuristic
    approximation.

    Args:
        linear_corr: Correlation matrix in linear space
        biomarker_types: List of distribution types ('lognormal', 'normal', or 'uniform')

    Returns:
        Approximate correlation matrix in transformed space

    Note:
        The most accurate approach is to calculate this from raw data.
        This is a reasonable approximation when raw data is unavailable.
    """
    n = len(biomarker_types)
    log_corr = linear_corr.copy()

    for i in range(n):
        for j in range(i + 1, n):
            # If both variables are log-normal, increase correlation slightly.
            if biomarker_types[i] == 'lognormal' and biomarker_types[j] == 'lognormal':
                # Heuristic: increase by 5%, but cap at 0.95.
                log_corr[i, j] = min(linear_corr[i, j] * 1.05, 0.95)
                log_corr[j, i] = log_corr[i, j]
            # If one is log-normal and one is normal, keep the linear correlation.

    return log_corr


def generate_initial_conditions(
    config: dict,
    sample_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    clip: bool = True
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate initial biomarker values from a configuration.

    If `config` includes a `correlation_matrix`, samples are drawn from a
    multivariate distribution (with log-space adjustment for log-normal
    biomarkers); otherwise, biomarkers are sampled independently from their
    marginal distributions.

    Args:
        config: Configuration dictionary from `load_ic_config`.
        sample_size: Number of samples (uses config default if None).
        random_seed: Random seed (uses config default if None).
        clip: Whether to clip values to each biomarker's range.

    Returns:
        biomarkers_array: Array of shape `(sample_size, n_biomarkers)`.
        biomarkers_dict: Mapping biomarker name → 1D array of samples.
    """
    # Set defaults
    if sample_size is None:
        sample_size = config['simulation_params']['default_sample_size']

    if random_seed is None:
        random_seed = config.get('random_seed', None)

    if random_seed is not None:
        np.random.seed(random_seed)

    biomarker_names = list(config['biomarkers'].keys())
    biomarkers_dict: Dict[str, np.ndarray] = {}

    # Check if correlation matrix is provided.
    correlation_matrix = config.get('correlation_matrix', None)

    if correlation_matrix is not None:
        # ---------------------------------------------------------------------
        # CORRELATED SAMPLING
        # ---------------------------------------------------------------------
        corr_linear = np.asarray(correlation_matrix, dtype=float)
        n_biomarkers = len(biomarker_names)

        if corr_linear.shape != (n_biomarkers, n_biomarkers):
            raise ValueError(
                f"'correlation_matrix' must be {n_biomarkers}x{n_biomarkers}, "
                f"got shape {corr_linear.shape}"
            )

        # Step 1: Collect marginal parameters in transformed space.
        mean_vector = []
        std_vector = []
        biomarker_types = []

        for name in biomarker_names:
            params = config['biomarkers'][name]
            dist = params.get('distribution', 'normal')
            biomarker_types.append(dist)

            if dist == 'lognormal':
                mean_linear = params['mean']
                std_linear = params['std']
                mu, sigma = calculate_lognormal_params(mean_linear, std_linear)
                mean_vector.append(mu)
                std_vector.append(sigma)
            elif dist == 'normal':
                mean_vector.append(params['mean'])
                std_vector.append(params['std'])
            elif dist == 'uniform':
                # For uniform variables, we model a standard normal in the
                # transformed space and later map it to a uniform distribution
                # via the normal CDF. Correlations are handled in the Gaussian
                # copula space.
                mean_vector.append(0.0)
                std_vector.append(1.0)
            else:
                raise ValueError(f"Unsupported distribution type: {dist}")

        mean_vector = np.array(mean_vector, dtype=float)
        std_vector = np.array(std_vector, dtype=float)

        # Step 2: Approximate correlation in transformed (log) space.
        corr_transformed = approximate_log_space_correlation(
            corr_linear,
            biomarker_types,
        )

        # Step 3: Build covariance matrix in transformed space.
        cov_matrix = corr_transformed * np.outer(std_vector, std_vector)

        # Step 4: Sample from multivariate normal in transformed space.
        try:
            samples_transformed = multivariate_normal.rvs(
                mean=mean_vector,
                cov=cov_matrix,
                size=sample_size,
            )
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "Covariance matrix is not positive semi-definite; "
                "cannot sample multivariate normal initial conditions."
            ) from e

        if samples_transformed.ndim == 1:
            samples_transformed = samples_transformed.reshape(1, -1)

        # Step 5: Transform back to original space.
        for j, name in enumerate(biomarker_names):
            params = config['biomarkers'][name]
            dist = biomarker_types[j]
            z = samples_transformed[:, j]

            if dist == 'lognormal':
                samples = np.exp(z)
            elif dist == 'uniform':
                # Map standard normal to U(0, 1) via CDF, then scale to
                # [clip_min, clip_max].
                u = norm.cdf(z)
                low = params['clip_min']
                high = params['clip_max']
                samples = low + u * (high - low)
            else:  # 'normal'
                samples = z

            if clip:
                samples = np.clip(samples, params['clip_min'], params['clip_max'])

            biomarkers_dict[name] = samples

        biomarkers_array = np.column_stack([biomarkers_dict[name] for name in biomarker_names])
        return biomarkers_array, biomarkers_dict

    # -------------------------------------------------------------------------
    # INDEPENDENT SAMPLING (no correlation matrix)
    # -------------------------------------------------------------------------
    for name, params in config['biomarkers'].items():
        distribution = params['distribution']

        if distribution == 'lognormal':
            mean = params['mean']
            std = params['std']
            mu, sigma = calculate_lognormal_params(mean, std)
            samples = np.random.lognormal(mean=mu, sigma=sigma, size=sample_size)
        elif distribution == 'normal':
            mean = params['mean']
            std = params['std']
            samples = np.random.normal(loc=mean, scale=std, size=sample_size)
        elif distribution == 'uniform':
            # Sample from a uniform distribution over the specified range.
            low = params.get('low', params['clip_min'])
            high = params.get('high', params['clip_max'])
            samples = np.random.uniform(low=low, high=high, size=sample_size)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution}")

        if clip:
            samples = np.clip(samples, params['clip_min'], params['clip_max'])

        biomarkers_dict[name] = samples

    biomarkers_array = np.column_stack([biomarkers_dict[name] for name in biomarker_names])
    return biomarkers_array, biomarkers_dict


def get_biomarker_order(config: dict) -> list:
    """Get the ordered list of biomarker names from config."""
    return list(config['biomarkers'].keys())


def get_observed_biomarker_order(config: dict) -> list:
    """Get the ordered list of *observed* biomarker names from config.

    A biomarker is considered observed if its config either omits the
    `observed` flag or sets it to True.
    """
    return [
        name
        for name, params in config['biomarkers'].items()
        if params.get('observed', True)
    ]


def get_trajectory_bin_width(config: dict, fallback: float = 14.0) -> float:
    """Return the trajectory bin width from config, or fallback if not set."""
    return float(config.get("simulation_params", {}).get("trajectory_bin_width", fallback))


def get_time_grid(config: dict, n_points: Optional[int] = None) -> np.ndarray:
    """Generate time grid for ODE simulation based on config.

    Args:
        config: Configuration dictionary
        n_points: Number of time points (uses config default if None)

    Returns:
        Array of time points

    Example:
        >>> config = load_ic_config('aki')
        >>> t = get_time_grid(config)
        >>> print(t)  # array([0, 1, 2, ..., 168])
    """
    sim_params = config['simulation_params']
    t_start, t_end = sim_params['time_span']

    if n_points is None:
        n_points = sim_params['time_points']

    return np.linspace(t_start, t_end, n_points)


def print_config_summary(config: dict):
    """Print a human-readable summary of the configuration.

    Args:
        config: Configuration dictionary
    """
    print(f"Problem: {config['problem_name']}")
    print(f"Description: {config['description']}")
    print(f"\nBiomarkers ({len(config['biomarkers'])}):")

    for name, params in config['biomarkers'].items():
        observed = params.get('observed', True)
        print(f"\n  {params['display_name']} ({name}):")
        print(f"    Unit: {params['unit']}")
        print(f"    Distribution: {params['distribution']}")
        mean_str = params.get('mean', 'N/A')
        std_str = params.get('std', 'N/A')
        print(f"    Mean: {mean_str}, Median: {params.get('median', 'N/A')}, Std: {std_str}")
        print(f"    Clip range: [{params['clip_min']}, {params['clip_max']}]")
        print(f"    Observed in data: {observed}")

    print(f"\nSimulation Parameters:")
    sim = config['simulation_params']
    print(f"  Time span: {sim['time_span']} {sim['time_unit']}")
    print(f"  Default sample size: {sim['default_sample_size']}")
    print(f"  Description: {sim['description']}")


def validate_config(config: dict) -> Tuple[bool, list]:
    """Validate configuration structure and required fields.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required top-level fields
    required_fields = ['problem_name', 'biomarkers', 'simulation_params']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Check biomarkers
    if 'biomarkers' in config:
        for name, params in config['biomarkers'].items():
            # Fields required for all biomarkers
            required_params = ['distribution', 'clip_min', 'clip_max']
            for param in required_params:
                if param not in params:
                    errors.append(f"Biomarker '{name}' missing required parameter: {param}")

            # Optional observed flag should be boolean if present.
            if 'observed' in params and not isinstance(params['observed'], bool):
                errors.append(f"Biomarker '{name}' has non-boolean 'observed' flag")
            # Optional negative flag should be boolean if present.
            if 'negative' in params and not isinstance(params['negative'], bool):
                errors.append(f"Biomarker '{name}' has non-boolean 'negative' flag")

            # Check distribution-specific requirements
            dist = params.get('distribution')
            if dist in ('normal', 'lognormal'):
                for param in ['mean', 'std']:
                    if param not in params:
                        errors.append(
                            f"Biomarker '{name}' with distribution '{dist}' missing required parameter: {param}"
                        )
            if dist == 'lognormal' and 'median' not in params:
                errors.append(f"Biomarker '{name}' uses lognormal but missing 'median'")

    # Check simulation params
    if 'simulation_params' in config:
        sim = config['simulation_params']
        required_sim = ['time_span', 'default_sample_size']
        for param in required_sim:
            if param not in sim:
                errors.append(f"Missing simulation parameter: {param}")

    return len(errors) == 0, errors
