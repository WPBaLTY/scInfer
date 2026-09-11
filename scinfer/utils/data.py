"""Data loading and validation utilities for AnnData objects.

Provides helper functions for loading single-cell data from various
formats and validating that AnnData objects meet the requirements
expected by scInfer's model adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from loguru import logger


def load_adata(
    path: Union[str, Path],
    backend: str = "auto",
) -> "AnnData":  # noqa: F821
    """Load an AnnData object from file.

    Supports ``.h5ad``, ``.loom``, and ``.csv`` formats.  The format is
    auto-detected from the file extension unless *backend* is specified.

    Parameters
    ----------
    path : str or Path
        Path to the data file.
    backend : str
        File format: ``"auto"``, ``"h5ad"``, ``"loom"``, ``"csv"``.

    Returns
    -------
    anndata.AnnData
        Loaded AnnData object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the format is unsupported.
    """
    try:
        import anndata
    except ImportError as exc:
        raise ImportError(
            "The 'anndata' package is required to load data files. "
            "Install it with: pip install anndata"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Auto-detect format from extension
    if backend == "auto":
        ext = path.suffix.lower()
        _EXT_MAP = {
            ".h5ad": "h5ad",
            ".loom": "loom",
            ".csv": "csv",
        }
        if ext not in _EXT_MAP:
            raise ValueError(
                f"Unsupported file extension {ext!r} for {path}. "
                f"Supported formats: {', '.join(_EXT_MAP.keys())}"
            )
        backend = _EXT_MAP[ext]

    logger.info("Loading AnnData from {} (backend={})", path, backend)

    _READERS = {
        "h5ad": anndata.read_h5ad,
        "loom": anndata.read_loom,
        "csv": anndata.read_csv,
    }

    if backend not in _READERS:
        raise ValueError(
            f"Unsupported backend {backend!r}. "
            f"Choose from: {', '.join(_READERS.keys())}"
        )

    adata = _READERS[backend](str(path))
    logger.info(
        "Loaded AnnData: {} cells × {} genes",
        adata.n_obs,
        adata.n_vars,
    )
    return adata


def validate_adata(
    adata: "AnnData",  # noqa: F821
    require_counts: bool = False,
    require_genes: Optional[list] = None,
) -> bool:
    """Validate that an AnnData object meets scInfer requirements.

    Checks:
    - ``adata.X`` is not ``None``.
    - If *require_counts* is ``True``, verifies that ``adata.raw`` or
      ``adata.layers["counts"]`` exists.
    - If *require_genes* is provided, checks that all gene names are present.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to validate.
    require_counts : bool
        Whether raw counts must be present.
    require_genes : list of str, optional
        Gene names that must be in ``adata.var_names``.

    Returns
    -------
    bool
        ``True`` if validation passes.

    Raises
    ------
    ValueError
        If validation fails.
    """
    # Check that adata.X is not None
    if adata.X is None:
        raise ValueError(
            "adata.X is None. The AnnData object must contain an expression matrix."
        )

    # Optionally check for raw counts
    if require_counts:
        has_raw = adata.raw is not None
        has_counts_layer = "counts" in adata.layers
        if not has_raw and not has_counts_layer:
            raise ValueError(
                "Raw counts are required but neither adata.raw nor "
                "adata.layers['counts'] is present. "
                "Provide raw (un-normalised) counts for proper model input."
            )

    # Optionally check required gene names
    if require_genes is not None and len(require_genes) > 0:
        missing = set(require_genes) - set(adata.var_names)
        if missing:
            raise ValueError(
                f"{len(missing)} required gene(s) not found in adata.var_names: "
                f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )

    logger.debug(
        "AnnData validation passed: {} cells × {} genes",
        adata.n_obs,
        adata.n_vars,
    )
    return True
