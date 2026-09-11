"""scinfer.adapters — Unified interface for single-cell foundation models.

Each adapter wraps a specific model (scGPT, Geneformer, scFoundation, UCE)
behind the common :class:`BaseModelAdapter` protocol so that optimisation
engines and the top-level API can work with any model interchangeably.

Available adapters
------------------
* :class:`ScGPTAdapter` — scGPT (53M params, Value Binning + dynamic mask)
* :class:`GeneformerAdapter` — Geneformer V2 (10M/38M/316M, Rank Embedding)
* :class:`ScFoundationAdapter` — scFoundation (121M, raw expression + Performer)
* :class:`UCEAdapter` — UCE (~650M, protein sequence embeddings)
"""

from scinfer.adapters.base import BaseModelAdapter
from scinfer.adapters.geneformer import GeneformerAdapter
from scinfer.adapters.scfoundation import ScFoundationAdapter
from scinfer.adapters.scgpt import ScGPTAdapter
from scinfer.adapters.uce import UCEAdapter

__all__ = [
    "BaseModelAdapter",
    "ScGPTAdapter",
    "GeneformerAdapter",
    "ScFoundationAdapter",
    "UCEAdapter",
]
