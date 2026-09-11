"""Knowledge distillation optimisation engine.

Provides knowledge distillation for single-cell foundation models, enabling
compression of large teacher models into smaller student models while
preserving biological representation quality.

Supported distillation strategies:
- ``representation``: Match intermediate representations (feature maps).
- ``task``: Match task-level output logits (classification / regression).
- ``contrastive``: Contrastive learning to align teacher-student embedding spaces.

If real training data is unavailable, the engine falls back to synthetic data
generation, ensuring the pipeline always runs end-to-end.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from scinfer.adapters.base import BaseModelAdapter
from scinfer.engines.base import BaseOptimizationEngine
from scinfer.core.registry import register_engine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DistillationConfig:
    """Knowledge distillation configuration.

    Parameters
    ----------
    strategy : str
        Distillation strategy: ``"representation"``, ``"task"``, or
        ``"contrastive"``.
    temperature : float
        Softmax temperature for logit distillation (task strategy).
    alpha : float
        Weight for the representation / task loss term.
    beta : float
        Weight for the contrastive loss term (contrastive strategy).
    lr : float
        Learning rate for student fine-tuning.
    n_epochs : int
        Number of distillation training epochs.
    batch_size : int
        Mini-batch size during distillation training.
    n_synthetic_samples : int
        Number of synthetic samples to generate when real data is unavailable.
    n_genes : int
        Number of genes (input dimension) for synthetic data.
    embed_dim : int
        Expected embedding dimension of the student model.
    contrastive_margin : float
        Margin for contrastive triplet loss.
    projection_dim : int
        Dimension of the projection head for contrastive learning.
    """

    strategy: str = "representation"
    temperature: float = 4.0
    alpha: float = 0.5
    beta: float = 0.3
    lr: float = 1e-4
    n_epochs: int = 10
    batch_size: int = 64
    n_synthetic_samples: int = 512
    n_genes: int = 2000
    embed_dim: int = 256
    contrastive_margin: float = 1.0
    projection_dim: int = 128

    def validate(self) -> bool:
        """Return ``True`` if the configuration is internally consistent."""
        if self.strategy not in ("representation", "task", "contrastive"):
            logger.error(f"Unknown distillation strategy: {self.strategy}")
            return False
        if self.temperature <= 0:
            logger.error("Temperature must be positive")
            return False
        if not (0.0 <= self.alpha <= 1.0):
            logger.error("alpha must be in [0, 1]")
            return False
        return True


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_samples: int,
    n_genes: int,
    n_cell_types: int = 5,
    device: torch.device = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic single-cell expression data.

    Each cell type has a distinct set of highly-expressed gene modules,
    mimicking real biological variation.

    Parameters
    ----------
    n_samples : int
        Number of cells to generate.
    n_genes : int
        Number of genes.
    n_cell_types : int
        Number of distinct cell types.
    device : torch.device, optional
        Target device.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(expression_matrix, labels)`` with shapes
        ``(n_samples, n_genes)`` and ``(n_samples,)``.
    """
    rng = np.random.RandomState(seed)
    device = device or torch.device("cpu")

    genes_per_type = n_genes // n_cell_types
    labels = rng.randint(0, n_cell_types, size=n_samples)

    # Baseline low-level expression + cell-type-specific signal
    data = rng.exponential(0.5, size=(n_samples, n_genes)).astype(np.float32)
    for ct in range(n_cell_types):
        mask = labels == ct
        start = ct * genes_per_type
        end = start + genes_per_type if ct < n_cell_types - 1 else n_genes
        data[mask, start:end] += rng.exponential(
            3.0, size=(mask.sum(), end - start)
        ).astype(np.float32)

    return (
        torch.tensor(data, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Projection head for contrastive learning
# ---------------------------------------------------------------------------

class _ProjectionHead(nn.Module):
    """Small MLP projection head for contrastive distillation.

    Parameters
    ----------
    input_dim : int
        Dimension of the teacher / student embeddings.
    projection_dim : int
        Output dimension of the projection.
    """

    def __init__(self, input_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DistillationEngine
# ---------------------------------------------------------------------------

@register_engine("distillation")
class DistillationEngine(BaseOptimizationEngine):
    """Inference optimisation engine: knowledge distillation.

    Compresses a large *teacher* model into a smaller *student* model by
    transferring knowledge through three interchangeable loss functions:

    * **Representation loss** – MSE / cosine similarity between intermediate
      feature maps.
    * **Task loss** – KL-divergence between softened output logits.
    * **Contrastive loss** – Triplet loss that pulls student embeddings
      closer to their teacher anchors while pushing apart different classes.

    When real training data is unavailable the engine automatically falls
    back to synthetic data generation.
    """

    SUPPORTED_STRATEGIES = ("representation", "task", "contrastive")

    def __init__(self) -> None:
        self._student_models: Dict[str, nn.Module] = {}
        self._configs: Dict[str, DistillationConfig] = {}
        self._training_history: Dict[str, List[float]] = {}
        logger.debug("DistillationEngine initialised")

    # ------------------------------------------------------------------
    # BaseOptimizationEngine interface
    # ------------------------------------------------------------------

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """Apply distillation to obtain an optimised (student) model.

        The *model* parameter is treated as the **teacher**.  A student
        architecture is created automatically (smaller copy) and trained
        via the configured distillation strategy.

        Parameters
        ----------
        model : torch.nn.Module
            Teacher model.
        config : dict
            Distillation configuration (see :class:`DistillationConfig`).

        Returns
        -------
        torch.nn.Module
            Trained student model.
        """
        distill_config = DistillationConfig(**{
            k: v for k, v in config.items()
            if k in DistillationConfig.__dataclass_fields__
        })

        if not distill_config.validate():
            raise ValueError(f"Invalid distillation config: {distill_config}")

        logger.info(
            f"Applying distillation: strategy={distill_config.strategy}, "
            f"temperature={distill_config.temperature}, alpha={distill_config.alpha}"
        )

        # Create student as a smaller copy
        student = self._create_student(model)

        # Generate or accept data
        data = config.get("data")
        if data is None:
            logger.info("No training data provided – using synthetic fallback")
            data, _ = _generate_synthetic_data(
                n_samples=distill_config.n_synthetic_samples,
                n_genes=distill_config.n_genes,
                device=next(model.parameters()).device
                if list(model.parameters())
                else torch.device("cpu"),
            )

        # Train
        student = self.train_distillation(
            teacher=model,
            student=student,
            data=data,
            config=distill_config,
        )

        student_id = f"distill_{distill_config.strategy}_{id(model) % 10000}"
        self._student_models[student_id] = student
        self._configs[student_id] = distill_config

        return student

    def benchmark(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, float]:
        """Benchmark the student model's inference performance.

        Parameters
        ----------
        model : torch.nn.Module
            Student (optimised) model.
        input_data : torch.Tensor
            Representative input tensor.

        Returns
        -------
        dict[str, float]
            Latency, throughput, and model-size metrics.
        """
        device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )
        if input_data.device != device:
            input_data = input_data.to(device)

        model.eval()

        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_data)

        # Latency
        n_runs = 50
        times: List[float] = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(input_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        latency_arr = np.array(times) * 1000  # ms
        throughput = 1000.0 / latency_arr.mean() if latency_arr.mean() > 0 else 0.0

        # Model size
        params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = params_bytes / (1024 ** 2)

        metrics = {
            "latency_ms": round(float(latency_arr.mean()), 4),
            "latency_std_ms": round(float(latency_arr.std()), 4),
            "latency_p50_ms": round(float(np.percentile(latency_arr, 50)), 4),
            "latency_p90_ms": round(float(np.percentile(latency_arr, 90)), 4),
            "latency_p99_ms": round(float(np.percentile(latency_arr, 99)), 4),
            "throughput_items_per_sec": round(float(throughput), 2),
            "model_size_mb": round(model_size_mb, 2),
            "n_parameters": sum(p.numel() for p in model.parameters()),
        }

        logger.info(
            f"Distillation benchmark: latency={metrics['latency_ms']:.3f} ms, "
            f"throughput={metrics['throughput_items_per_sec']:.1f} it/s, "
            f"size={metrics['model_size_mb']:.1f} MB"
        )
        return metrics

    def is_compatible(self, adapter: BaseModelAdapter) -> bool:
        """Check whether distillation is compatible with the given adapter.

        Parameters
        ----------
        adapter : BaseModelAdapter
            Model adapter to check.

        Returns
        -------
        bool
            ``True`` if the adapter supports distillation or is an
            encoder-based architecture.
        """
        supported = getattr(adapter, "supported_optimizations", [])
        if "distillation" in supported:
            return True
        # Encoder-based models are generally compatible
        model_type = getattr(adapter, "model_type", "")
        return "encoder" in model_type

    @property
    def engine_name(self) -> str:
        return "distillation"

    @property
    def optimization_type(self) -> str:
        return "distillation"

    def validate(self, model: nn.Module, config: Dict[str, Any]) -> bool:
        """Validate model and config for distillation compatibility.

        Parameters
        ----------
        model : torch.nn.Module
            The model to validate (treated as teacher).
        config : dict
            Distillation configuration values.

        Returns
        -------
        bool
            ``True`` if the configuration is valid and the model can be used
            for distillation.
        """
        try:
            distill_config = DistillationConfig(**{
                k: v for k, v in config.items()
                if k in DistillationConfig.__dataclass_fields__
            })
            return distill_config.validate()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Distillation-specific methods
    # ------------------------------------------------------------------

    def train_distillation(
        self,
        teacher: nn.Module,
        student: nn.Module,
        data: torch.Tensor,
        config: DistillationConfig,
    ) -> nn.Module:
        """Run the distillation training loop.

        Parameters
        ----------
        teacher : torch.nn.Module
            Pre-trained teacher model (frozen).
        student : torch.nn.Module
            Student model to train.
        data : torch.Tensor
            Training data (expression matrix).
        config : DistillationConfig
            Distillation hyper-parameters.

        Returns
        -------
        torch.nn.Module
            Trained student model.
        """
        device = (
            next(teacher.parameters()).device
            if list(teacher.parameters())
            else torch.device("cpu")
        )
        student = student.to(device)
        teacher.eval()

        # Freeze teacher
        for p in teacher.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam(student.parameters(), lr=config.lr)
        n_samples = data.shape[0]

        history: List[float] = []

        for epoch in range(config.n_epochs):
            student.train()
            epoch_losses: List[float] = []

            # Mini-batch iteration
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, config.batch_size):
                idx = perm[start : start + config.batch_size]
                batch = data[idx]

                with torch.no_grad():
                    teacher_out = teacher(batch)

                student_out = student(batch)

                # Align dimensions if necessary
                teacher_out, student_out = self._align_dimensions(teacher_out, student_out)

                # Compute loss based on strategy
                if config.strategy == "representation":
                    loss = self._representation_loss(teacher_out, student_out)
                elif config.strategy == "task":
                    loss = self._task_loss(
                        teacher_out, student_out, temperature=config.temperature
                    )
                elif config.strategy == "contrastive":
                    loss = self._contrastive_loss(
                        teacher_out, student_out, margin=config.contrastive_margin
                    )
                else:
                    raise ValueError(f"Unknown strategy: {config.strategy}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            history.append(mean_loss)

            if (epoch + 1) % max(1, config.n_epochs // 5) == 0 or epoch == 0:
                logger.info(
                    f"  Distillation epoch {epoch + 1}/{config.n_epochs} – "
                    f"loss={mean_loss:.6f}"
                )

        self._training_history[f"distill_{config.strategy}"] = history
        student.eval()
        return student

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    @staticmethod
    def _representation_loss(
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
    ) -> torch.Tensor:
        """MSE + cosine similarity loss between teacher and student representations.

        Parameters
        ----------
        teacher_features : torch.Tensor
            Teacher intermediate representations.
        student_features : torch.Tensor
            Student intermediate representations (same shape after alignment).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Normalise to unit vectors for cosine loss
        t_norm = F.normalize(teacher_features, dim=-1)
        s_norm = F.normalize(student_features, dim=-1)
        cosine_loss = 1.0 - (t_norm * s_norm).sum(dim=-1).mean()

        # MSE loss
        mse_loss = F.mse_loss(student_features, teacher_features)

        return 0.5 * cosine_loss + 0.5 * mse_loss

    @staticmethod
    def _task_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float = 4.0,
    ) -> torch.Tensor:
        """KL-divergence between softened teacher and student logits.

        Parameters
        ----------
        teacher_logits : torch.Tensor
            Teacher output logits.
        student_logits : torch.Tensor
            Student output logits.
        temperature : float
            Softmax temperature.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        T = temperature
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        student_log_soft = F.log_softmax(student_logits / T, dim=-1)
        kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (T ** 2)
        return kl

    @staticmethod
    def _contrastive_loss(
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Simplified contrastive loss treating teacher embeddings as anchors.

        Uses a triplet-style formulation:
        - *anchor*: teacher embedding
        - *positive*: corresponding student embedding
        - *negative*: a randomly shuffled student embedding

        Parameters
        ----------
        teacher_embeddings : torch.Tensor
            Teacher output embeddings ``(batch, dim)``.
        student_embeddings : torch.Tensor
            Student output embeddings ``(batch, dim)``.
        margin : float
            Triplet loss margin.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Positive distance: teacher ↔ own student
        pos_dist = F.pairwise_distance(teacher_embeddings, student_embeddings, p=2)

        # Negative: teacher ↔ shuffled student
        shuffled_idx = torch.randperm(teacher_embeddings.shape[0], device=teacher_embeddings.device)
        neg_dist = F.pairwise_distance(
            teacher_embeddings, student_embeddings[shuffled_idx], p=2
        )

        # Triplet loss
        losses = F.relu(pos_dist - neg_dist + margin)
        return losses.mean()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_student(teacher: nn.Module) -> nn.Module:
        """Create a smaller student model by deep-copying and shrinking layers.

        The student keeps the same architecture but with halved hidden
        dimensions (minimum 32) and halved number of transformer layers
        (minimum 1).

        Parameters
        ----------
        teacher : torch.nn.Module
            Teacher model.

        Returns
        -------
        torch.nn.Module
            Student model (deep copy with reduced capacity).
        """
        student = copy.deepcopy(teacher)

        # Halve Linear layer dimensions
        for module in student.modules():
            if isinstance(module, nn.Linear):
                module.in_features = max(32, module.in_features // 2)
                module.out_features = max(32, module.out_features // 2)
                # Re-initialise weights to match new shape
                module.weight = nn.Parameter(
                    torch.empty(module.out_features, module.in_features)
                )
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias = nn.Parameter(torch.zeros(module.out_features))

        # Reduce TransformerEncoder layers
        for module in student.modules():
            if isinstance(module, nn.TransformerEncoder):
                current_layers = len(module.layers)
                target_layers = max(1, current_layers // 2)
                module.layers = module.layers[:target_layers]

        # Re-initialise all remaining parameters
        for p in student.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                nn.init.normal_(p, std=0.02)

        return student

    @staticmethod
    def _align_dimensions(
        teacher_out: torch.Tensor,
        student_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align teacher and student output dimensions if they differ.

        Uses zero-padding or truncation to match the last dimension.

        Parameters
        ----------
        teacher_out : torch.Tensor
            Teacher output.
        student_out : torch.Tensor
            Student output.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Dimension-aligned ``(teacher_out, student_out)``.
        """
        if teacher_out.shape == student_out.shape:
            return teacher_out, student_out

        # Match all dimensions
        t_shape = list(teacher_out.shape)
        s_shape = list(student_out.shape)

        # Pad or truncate each dimension
        for dim in range(max(len(t_shape), len(s_shape))):
            t_size = t_shape[dim] if dim < len(t_shape) else 1
            s_size = s_shape[dim] if dim < len(s_shape) else 1
            if t_size != s_size:
                min_size = min(t_size, s_size)
                slices_t = [slice(None)] * len(t_shape)
                slices_s = [slice(None)] * len(s_shape)
                slices_t[dim] = slice(0, min_size)
                slices_s[dim] = slice(0, min_size)
                teacher_out = teacher_out[tuple(slices_t)]
                student_out = student_out[tuple(slices_s)]

        return teacher_out, student_out

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_student_model(self, student_id: str) -> Optional[nn.Module]:
        """Retrieve a previously trained student model.

        Parameters
        ----------
        student_id : str
            Student identifier.

        Returns
        -------
        torch.nn.Module or None
        """
        return self._student_models.get(student_id)

    def get_training_history(self, key: str) -> List[float]:
        """Return the training loss history for a given distillation run.

        Parameters
        ----------
        key : str
            Training history key (e.g. ``"distill_representation"``).

        Returns
        -------
        list[float]
            Per-epoch loss values.
        """
        return self._training_history.get(key, [])
