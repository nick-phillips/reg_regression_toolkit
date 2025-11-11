"""Generate the dummy expression and metadata tables used by the tests/examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(21)

CLASS_SPECS = [
    ("BLCA", "BLCA", 8, np.array([1.2, 0.8, 0.6]), np.array([0.45, 0.4, 0.45])),
    ("CTRL", "Control", 8, np.array([-0.4, -0.3, -0.2]), np.array([0.8, 0.75, 0.7])),
    ("OTHER", "Other", 8, np.array([0.2, 0.25, 0.3]), np.array([0.55, 0.45, 0.5])),
]


def _generate_block(
    prefix: str,
    label: str,
    count: int,
    mean_vec: np.ndarray,
    std_vec: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = [f"{prefix}_{idx:02d}" for idx in range(1, count + 1)]
    base = RNG.normal(loc=mean_vec, scale=std_vec, size=(count, 3))

    primary = base[:, 0]
    partner_seed = base[:, 1]
    secondary = base[:, 2]
    partner = 0.7 * primary + 0.3 * partner_seed + RNG.normal(0.0, 0.25, size=count)

    noise = RNG.normal(0.0, 1.0, size=(count, 3))

    expression = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "feature_signal_primary": primary,
            "feature_signal_partner": partner,
            "feature_signal_secondary": secondary,
            "feature_noise_a": noise[:, 0],
            "feature_noise_b": noise[:, 1],
            "feature_noise_c": noise[:, 2],
        }
    )

    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "Sample_type": [label] * count,
        }
    )
    return expression, metadata


def main() -> None:
    expression_frames: list[pd.DataFrame] = []
    metadata_frames: list[pd.DataFrame] = []

    for prefix, label, count, means, stds in CLASS_SPECS:
        expr, meta = _generate_block(prefix, label, count, means, stds)
        expression_frames.append(expr)
        metadata_frames.append(meta)

    expression = pd.concat(expression_frames, ignore_index=True)
    metadata = pd.concat(metadata_frames, ignore_index=True)

    expression.to_csv(DATA_DIR / "dummy_expression.csv", index=False)
    metadata.to_csv(DATA_DIR / "dummy_metadata.csv", index=False)


if __name__ == "__main__":
    main()
