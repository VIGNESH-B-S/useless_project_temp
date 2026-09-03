"""Convert measured pen-path length to ink use with a known pen rate."""

from dataclasses import asdict, dataclass

DEFAULT_ML_PER_CM = 0.000005


@dataclass(frozen=True)
class InkEstimate:
    """A reproducible fixed-rate ink estimate, including its assumption."""

    path_length_mm: float
    rate_ml_per_cm: float
    estimated_ink_ml: float
    estimated_ink_microlitres: float

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_ink_ml(length_mm: float, ml_per_cm: float = DEFAULT_ML_PER_CM) -> float:
    """Return estimated ink use in mL for a trajectory length in millimetres."""
    if length_mm < 0:
        raise ValueError("length_mm cannot be negative")
    if ml_per_cm < 0:
        raise ValueError("ml_per_cm cannot be negative")
    return (length_mm / 10.0) * ml_per_cm


def estimate_ink(length_mm: float, ml_per_cm: float = DEFAULT_ML_PER_CM) -> InkEstimate:
    """Return both mL and microlitres from a trajectory length in millimetres."""
    volume_ml = estimate_ink_ml(length_mm, ml_per_cm)
    return InkEstimate(
        path_length_mm=length_mm,
        rate_ml_per_cm=ml_per_cm,
        estimated_ink_ml=volume_ml,
        estimated_ink_microlitres=volume_ml * 1000.0,
    )
