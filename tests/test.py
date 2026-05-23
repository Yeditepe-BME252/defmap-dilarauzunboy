from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Library import case_24


def test_case_24_mapping_matches_assigned_parameters():
    x = np.array([[-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]])
    y = np.array([[-2.0, -2.0, -2.0], [1.0, 1.0, 1.0]])

    x_deformed, y_deformed = case_24(x, y)

    expected_x = x + 0.1 * np.sin(2 * np.pi * y / 4)
    assert np.allclose(x_deformed, expected_x)
    assert np.allclose(y_deformed, y)


def test_case_24_is_transverse_wave_in_x_direction():
    x = np.array([0.0, 0.0, 0.0])
    y = np.array([-2.0, 0.0, 1.0])

    x_deformed, y_deformed = case_24(x, y)

    assert np.allclose(x_deformed - x, [0.0, 0.0, 0.1])
    assert np.allclose(y_deformed, y)


def test_grid_output_exists():
    assert Path("grid.png").is_file()
