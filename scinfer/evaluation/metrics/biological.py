"""Biological quality metrics.

Provides the :class:`BiologicalMetrics` class for 6-dimensional biological
validation of optimised single-cell models:

1. Cell-type annotation (F1, ARI, NMI)
2. Embedding-space fidelity (kNN accuracy, trustworthiness, continuity)
3. Perturbation prediction (PCC, Spearman correlation)
4. Gene-network inference (AUROC, AUPRC)
5. Batch-effect correction (ASW)
6. Differential-expression consistency (Jaccard index)

Also exposes convenience functions ``compute_ari``, ``compute_nmi``,
``compute_f1``, and ``compute_pcc`` for quick one-shot measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Optional imports — gracefully degrade if unavailable
# ---------------------------------------------------------------------------

try:
    from sklearn.metrics import (
        adjusted_rand_score,
        f1_score,
        normalized_mutual_info_score,
        silhouette_score,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    from scipy.stats import pearsonr, spearmanr
    _SCIPY = True
except ImportError:
    _SCIPY = False


def _require(pkg: str) -> None:
    if not (_SKLEARN if pkg == "sklearn" else _SCIPY):
        raise ImportError(f"{pkg} is required for this metric but is not installed.")


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class CellTypeResult:
    """Cell-type annotation metrics."""
    ari: float = 0.0
    nmi: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"ari": self.ari, "nmi": self.nmi,
                "f1_macro": self.f1_macro, "f1_weighted": self.f1_weighted}


@dataclass
class EmbeddingFidelityResult:
    """Embedding-space fidelity metrics."""
    knn_accuracy: float = 0.0
    trustworthiness: float = 0.0
    continuity: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"knn_accuracy": self.knn_accuracy,
                "trustworthiness": self.trustworthiness,
                "continuity": self.continuity}


@dataclass
class PerturbationResult:
    """Perturbation prediction metrics."""
    pcc: float = 0.0
    pcc_pvalue: float = 0.0
    spearman_rho: float = 0.0
    spearman_pvalue: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"pcc": self.pcc, "pcc_pvalue": self.pcc_pvalue,
                "spearman_rho": self.spearman_rho, "spearman_pvalue": self.spearman_pvalue}


@dataclass
class GeneNetworkResult:
    """Gene-network inference metrics."""
    auroc: float = 0.0
    auprc: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"auroc": self.auroc, "auprc": self.auprc}


@dataclass
class BatchCorrectionResult:
    """Batch-effect correction metrics."""
    asw_batch: float = 0.0
    asw_label: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"asw_batch": self.asw_batch, "asw_label": self.asw_label}


@dataclass
class DEResult:
    """Differential expression consistency metrics."""
    jaccard_topn: float = 0.0
    jaccard_per_comparison: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        return {"jaccard_topn": self.jaccard_topn}


# ---------------------------------------------------------------------------
# BiologicalMetrics — main class
# ---------------------------------------------------------------------------

class BiologicalMetrics:
    """Evaluate biological fidelity of optimised models.

    Six-dimensional biological validation:
    1. Cell-type annotation: F1, ARI, NMI
    2. Embedding-space fidelity: kNN accuracy, trustworthiness, continuity
    3. Perturbation prediction: PCC, Spearman correlation
    4. Gene-network inference: AUROC, AUPRC
    5. Batch-effect correction: ASW (Average Silhouette Width)
    6. Differential-expression consistency: Jaccard index (top-N DE genes)
    """

    # ------------------------------------------------------------------
    # 1. Cell-type annotation
    # ------------------------------------------------------------------

    @staticmethod
    def cell_type_annotation(
        embeddings: np.ndarray,
        labels: np.ndarray,
        *,
        cluster_labels: Optional[np.ndarray] = None,
    ) -> CellTypeResult:
        """Compute cell-type annotation metrics.

        Parameters
        ----------
        embeddings : np.ndarray
            Model embeddings (n_cells, embed_dim).
        labels : np.ndarray
            Ground-truth cell-type labels (n_cells,).
        cluster_labels : np.ndarray, optional
            Predicted cluster labels.  If ``None``, a simple k-means
            clustering is performed on *embeddings* (requires sklearn).

        Returns
        -------
        CellTypeResult
            ARI, NMI, macro-F1, weighted-F1.
        """
        _require("sklearn")

        if cluster_labels is None:
            from sklearn.cluster import KMeans
            n_clusters = len(np.unique(labels))
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            cluster_labels = km.fit_predict(embeddings)

        ari = float(adjusted_rand_score(labels, cluster_labels))
        nmi = float(normalized_mutual_info_score(labels, cluster_labels))
        f1_macro = float(f1_score(labels, cluster_labels, average="macro", zero_division=0))
        f1_weighted = float(f1_score(labels, cluster_labels, average="weighted", zero_division=0))

        result = CellTypeResult(ari=ari, nmi=nmi, f1_macro=f1_macro, f1_weighted=f1_weighted)
        logger.debug(f"Cell-type: ARI={ari:.4f}, NMI={nmi:.4f}, F1_macro={f1_macro:.4f}")
        return result

    # ------------------------------------------------------------------
    # 2. Embedding fidelity
    # ------------------------------------------------------------------

    @staticmethod
    def embedding_fidelity(
        original_emb: np.ndarray,
        optimized_emb: np.ndarray,
        k: int = 15,
    ) -> EmbeddingFidelityResult:
        """Evaluate embedding-space structural preservation.

        Parameters
        ----------
        original_emb : np.ndarray
            Baseline embeddings (n_cells, dim).
        optimized_emb : np.ndarray
            Optimised embeddings (n_cells, dim).
        k : int
            Number of nearest neighbours.

        Returns
        -------
        EmbeddingFidelityResult
            kNN accuracy, trustworthiness, continuity.
        """
        _require("sklearn")

        n = original_emb.shape[0]
        k = min(k, n - 1)

        # Build KNN graphs
        nn_orig = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
        nn_orig.fit(original_emb)
        _, idx_orig = nn_orig.kneighbors(original_emb)
        idx_orig = idx_orig[:, 1:]  # exclude self

        nn_opt = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
        nn_opt.fit(optimized_emb)
        _, idx_opt = nn_opt.kneighbors(optimized_emb)
        idx_opt = idx_opt[:, 1:]

        # kNN accuracy: fraction of original KNN preserved in optimised
        matches = 0
        total = 0
        for i in range(n):
            orig_set = set(idx_orig[i].tolist())
            opt_set = set(idx_opt[i].tolist())
            matches += len(orig_set & opt_set)
            total += k
        knn_accuracy = matches / max(total, 1)

        # Trustworthiness: for each point, penalise optimised-near neighbours
        # that are far in original space
        from scipy.spatial.distance import cdist

        # Ranks in original space
        dist_orig = cdist(original_emb, original_emb, metric="euclidean")
        rank_orig = np.argsort(np.argsort(dist_orig, axis=1), axis=1)

        # Trustworthiness
        trust_sum = 0.0
        for i in range(n):
            opt_neighbors = idx_opt[i]
            for j in opt_neighbors:
                if j == i:
                    continue
                r_orig = rank_orig[i, j]
                if r_orig > k:
                    trust_sum += (r_orig - k)
        trust_denom = n * k * (2 * n - 3 * k - 1) / 2.0
        trustworthiness = 1.0 - (2.0 / max(trust_denom, 1.0)) * trust_sum

        # Continuity: for each point, penalise original-near neighbours
        # that are far in optimised space
        dist_opt = cdist(optimized_emb, optimized_emb, metric="euclidean")
        rank_opt = np.argsort(np.argsort(dist_opt, axis=1), axis=1)

        cont_sum = 0.0
        for i in range(n):
            orig_neighbors = idx_orig[i]
            for j in orig_neighbors:
                if j == i:
                    continue
                r_opt = rank_opt[i, j]
                if r_opt > k:
                    cont_sum += (r_opt - k)
        cont_denom = n * k * (2 * n - 3 * k - 1) / 2.0
        continuity = 1.0 - (2.0 / max(cont_denom, 1.0)) * cont_sum

        result = EmbeddingFidelityResult(
            knn_accuracy=knn_accuracy,
            trustworthiness=max(0.0, min(1.0, trustworthiness)),
            continuity=max(0.0, min(1.0, continuity)),
        )
        logger.debug(
            f"Embedding fidelity: kNN={knn_accuracy:.4f}, "
            f"trust={trustworthiness:.4f}, cont={continuity:.4f}"
        )
        return result

    # ------------------------------------------------------------------
    # 3. Perturbation prediction
    # ------------------------------------------------------------------

    @staticmethod
    def perturbation_prediction(
        pred: np.ndarray,
        truth: np.ndarray,
    ) -> Union[PerturbationResult, List[PerturbationResult]]:
        """Evaluate perturbation prediction accuracy.

        Parameters
        ----------
        pred : np.ndarray
            Predicted expression values.  Shape ``(n_genes,)`` for a
            single condition or ``(n_conditions, n_genes)`` for batched.
        truth : np.ndarray
            Ground-truth expression values with the same shape as *pred*.

        Returns
        -------
        PerturbationResult or list[PerturbationResult]
            PCC and Spearman correlation per condition.
        """
        _require("scipy")

        pred = np.asarray(pred, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)

        if pred.ndim == 1:
            return BiologicalMetrics._perturbation_single(pred, truth)

        # Batched: (n_conditions, n_genes)
        results = []
        for p_row, t_row in zip(pred, truth):
            results.append(BiologicalMetrics._perturbation_single(p_row, t_row))
        return results

    @staticmethod
    def _perturbation_single(pred: np.ndarray, truth: np.ndarray) -> PerturbationResult:
        pcc_val, pcc_p = pearsonr(pred, truth)
        sp_val, sp_p = spearmanr(pred, truth)
        return PerturbationResult(
            pcc=float(pcc_val), pcc_pvalue=float(pcc_p),
            spearman_rho=float(sp_val), spearman_pvalue=float(sp_p),
        )

    # ------------------------------------------------------------------
    # 4. Gene-network inference
    # ------------------------------------------------------------------

    @staticmethod
    def gene_network_inference(
        predicted_adj: np.ndarray,
        true_adj: np.ndarray,
    ) -> GeneNetworkResult:
        """Evaluate gene regulatory network inference.

        Parameters
        ----------
        predicted_adj : np.ndarray
            Predicted adjacency matrix or flattened edge scores (n_edges,).
        true_adj : np.ndarray
            Ground-truth binary adjacency matrix or flattened labels.

        Returns
        -------
        GeneNetworkResult
            AUROC and AUPRC.
        """
        _require("sklearn")

        y_score = np.asarray(predicted_adj, dtype=np.float64).ravel()
        y_true = np.asarray(true_adj, dtype=np.float64).ravel()

        # Binarise ground truth
        y_true_bin = (y_true > 0.5).astype(int)

        auroc = float(roc_auc_score(y_true_bin, y_score))
        auprc = float(average_precision_score(y_true_bin, y_score))

        result = GeneNetworkResult(auroc=auroc, auprc=auprc)
        logger.debug(f"Gene network: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
        return result

    # ------------------------------------------------------------------
    # 5. Batch-effect correction
    # ------------------------------------------------------------------

    @staticmethod
    def batch_effect_correction(
        embeddings: np.ndarray,
        batch_labels: np.ndarray,
        *,
        cell_type_labels: Optional[np.ndarray] = None,
    ) -> BatchCorrectionResult:
        """Evaluate batch-effect correction quality.

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings (n_cells, dim).
        batch_labels : np.ndarray
            Batch assignment per cell.
        cell_type_labels : np.ndarray, optional
            Cell-type labels (needed for biology-aware ASW).

        Returns
        -------
        BatchCorrectionResult
            ASW for batch mixing (lower is better → stored as 1 - ASW_batch)
            and ASW for cell-type separation (higher is better).
        """
        _require("sklearn")

        embeddings = np.asarray(embeddings, dtype=np.float64)
        batch_labels = np.asarray(batch_labels)

        # Batch ASW — lower silhouette means better mixing
        unique_batches = np.unique(batch_labels)
        if len(unique_batches) < 2 or len(unique_batches) >= len(batch_labels):
            asw_batch = 0.0
        else:
            asw_batch_raw = float(silhouette_score(embeddings, batch_labels))
            # Convert so that 1 = perfect mixing, 0 = no mixing
            asw_batch = 1.0 - abs(asw_batch_raw)

        asw_label = 0.0
        if cell_type_labels is not None:
            cell_type_labels = np.asarray(cell_type_labels)
            unique_ct = np.unique(cell_type_labels)
            if 1 < len(unique_ct) < len(cell_type_labels):
                asw_label = float(silhouette_score(embeddings, cell_type_labels))

        result = BatchCorrectionResult(asw_batch=asw_batch, asw_label=asw_label)
        logger.debug(f"Batch correction: ASW_batch={asw_batch:.4f}, ASW_label={asw_label:.4f}")
        return result

    # ------------------------------------------------------------------
    # 6. Differential expression consistency
    # ------------------------------------------------------------------

    @staticmethod
    def differential_expression_consistency(
        de_genes_original: Union[np.ndarray, List[List[str]]],
        de_genes_optimized: Union[np.ndarray, List[List[str]]],
        top_n: int = 50,
    ) -> DEResult:
        """Evaluate differential-expression gene-list consistency.

        Parameters
        ----------
        de_genes_original : array-like or list of list of str
            Top-N DE gene identifiers from the baseline model.
            Can be 1-D (single comparison) or 2-D (multiple comparisons).
        de_genes_optimized : array-like or list of list of str
            Top-N DE gene identifiers from the optimised model.
        top_n : int
            Number of top genes to consider for Jaccard overlap.

        Returns
        -------
        DEResult
            Jaccard index for top-N DE genes.
        """
        def _to_set_list(x: Union[np.ndarray, List]) -> List[set]:
            if isinstance(x, np.ndarray):
                if x.ndim == 1:
                    return [set(x[:top_n].tolist())]
                return [set(row[:top_n].tolist()) for row in x]
            # list of lists
            if len(x) == 0:
                return []
            if isinstance(x[0], (list, tuple, np.ndarray)):
                return [set(list(row)[:top_n]) for row in x]
            return [set(x[:top_n])]

        sets_orig = _to_set_list(de_genes_original)
        sets_opt = _to_set_list(de_genes_optimized)

        jaccards: List[float] = []
        for s_o, s_p in zip(sets_orig, sets_opt):
            union = s_o | s_p
            if len(union) == 0:
                jaccards.append(0.0)
            else:
                jaccards.append(len(s_o & s_p) / len(union))

        mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0
        result = DEResult(jaccard_topn=mean_jaccard, jaccard_per_comparison=jaccards)
        logger.debug(f"DE consistency: Jaccard(top{top_n})={mean_jaccard:.4f}")
        return result


# ---------------------------------------------------------------------------
# Convenience functions (backward-compatible with placeholder signatures)
# ---------------------------------------------------------------------------

def compute_ari(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """Compute Adjusted Rand Index (ARI)."""
    _require("sklearn")
    return float(adjusted_rand_score(labels_true, labels_pred))


def compute_nmi(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
    average_method: str = "arithmetic",
) -> float:
    """Compute Normalised Mutual Information (NMI)."""
    _require("sklearn")
    return float(normalized_mutual_info_score(labels_true, labels_pred))


def compute_f1(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute F1 score for cell-type annotation transfer."""
    _require("sklearn")
    return float(f1_score(labels_true, labels_pred, average=average, zero_division=0))


def compute_pcc(
    embeddings_optimized: np.ndarray,
    embeddings_baseline: np.ndarray,
) -> float:
    """Compute mean Pearson Correlation Coefficient between embedding spaces.

    Computes PCC along the first axis (per-cell or per-dimension) and
    returns the mean.
    """
    _require("scipy")
    opt = np.asarray(embeddings_optimized, dtype=np.float64)
    base = np.asarray(embeddings_baseline, dtype=np.float64)

    if opt.ndim == 1:
        r, _ = pearsonr(opt, base)
        return float(r)

    # Per-dimension PCC
    pccs = []
    for d in range(opt.shape[1]):
        r, _ = pearsonr(opt[:, d], base[:, d])
        pccs.append(r)
    return float(np.mean(pccs))
