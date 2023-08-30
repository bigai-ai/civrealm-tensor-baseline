"""Base logger."""

import time
import os
import numpy as np


class BaseLogger:
    """Base logger class.
    Used for logging information in the on-policy training pipeline.
    """

    def __init__(self, args, algo_args, env_args, writter, run_dir):
        """Initialize the logger."""
        pass

    def close(self):
        """Close the logger."""
        self.log_file.close()
