"""
Configuration Loader

Inputs:
    - config.yaml: The master configuration file containing all dynamic parameters (Prometheus URLs, paths, training hyperparameters).

Outputs:
    - A globally accessible Python dictionary containing the parsed YAML configuration.

Process:
    1. Implements a Singleton pattern to ensure the YAML file is only read and parsed once from disk.
    2. Provides helper functions for other scripts to fetch the config object safely without duplicate I/O operations.
"""
import os
import yaml

_CONFIG = None

def load_config(path=None):
    """
    Purpose: Reads and parses the config.yaml file from disk into a Python dictionary, caching it in memory for future access.
    """
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(path) as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG

def get_config():
    """
    Purpose: Acts as the primary access point for other scripts to retrieve the configuration. Loads it if it hasn't been loaded yet.
    """
    if _CONFIG is None:
        return load_config()
    return _CONFIG
