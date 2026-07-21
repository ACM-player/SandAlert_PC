from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_tdoa import circular_absolute_error


def test_circular_absolute_error_handles_wrap_around():
    assert circular_absolute_error(359.0, 1.0) == 2.0
    assert circular_absolute_error(1.0, 359.0) == 2.0
    assert circular_absolute_error(0.0, 180.0) == 180.0


def test_circular_absolute_error_supports_arrays():
    errors = circular_absolute_error(
        np.array([359.0, 1.0, 45.0]),
        np.array([1.0, 359.0, 40.0]),
    )
    np.testing.assert_allclose(errors, [2.0, 2.0, 5.0])
