"""
pipeline.py - Synthetic Data Pipeline for severely imbalanced (binary) datasets.

Leakage-prevention contract (enforced by construction):
    1. Stratified train/validation/test split happens FIRST.
    2. Imputer + scaler are fitted on the training-fit partition ONLY.
    3. Resampling is applied to the training-fit partition ONLY. Validation and
       test partitions are never resampled.
    4. The CNN backbone is a frozen, pretrained network (no parameters are learned
       from this dataset), so extracting image features before the split is safe.
    5. The decision threshold is tuned on the validation partition by default.
       Tuning on the test set is available but is flagged as optimistic.

Components:
    CNNFeatureExtractor    - ResNet backbone -> latent vectors for images
    DataPreparer           - stratified splitting + train-only preprocessing
    RAGStrategyEngine      - Qdrant lookup of hyperparameter / compliance guidance
    AgenticOrchestrator    - LangGraph-style state machine with self-healing retries
    HonestEvaluator        - Precision / Recall / F1 / PR-AUC, threshold optimizer
    SyntheticDataPipeline  - facade used by the FastAPI layer
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms

logger = logging.getLogger("synthetic_data_pipeline")

SUPPORTED_STRATEGIES = ("smote", "adasyn", "borderline_smote", "smote_tomek", "random_over")


# --------------------------------------------------------------------------- #
# Errors and configuration
# --------------------------------------------------------------------------- #
class PipelineError(Exception):
    """Base class for all pipeline errors."""


class DataValidationError(PipelineError):
    """Raised when input data cannot be used safely."""


class ResamplingError(PipelineError):
    """Raised when a resampler fails."""


class TrainingError(PipelineError):
    """Raised when model training fails."""


class StageOrderError(PipelineError):
    """Raised when a stage is called before its prerequisites."""


@dataclass
class PipelineConfig:
    """Tunable settings. Everything has a safe default."""

    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    classifier: str = "logreg"            # "logreg" | "rf"
    threshold_source: str = "validation"  # "validation" | "test" (test is optimistic)
    max_retries: int = 3
    min_class_samples: int = 10
    noise_neighbors: int = 5
    noise_sample_cap: int = 20000
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = "imbalance_knowledge"
    embedding_dim: int = 256
    cnn_pretrained: bool = True
    image_size: int = 224
    batch_size: int = 32


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class SplitData:
    """Preprocessed partitions. Label 1 always means the minority class."""

    X_fit: np.ndarray
    y_fit: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    preprocessor: Pipeline
    minority_label: Any
    modality: str


@dataclass
class DatasetProfile:
    """Properties the agent routes on."""

    modality: str
    n_samples: int
    n_features: int
    n_minority: int
    n_majority: int
    minority_fraction: float
    imbalance_ratio: float
    noise_level: float  # share of minority points whose nearest neighbours are all majority

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# CNN feature extractor (unstructured image data)
# --------------------------------------------------------------------------- #
class CNNFeatureExtractor:
    """Frozen ResNet18 backbone that maps images to 512-d latent vectors."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_backbone(cfg.cnn_pretrained)
        self.transform = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _build_backbone(self, pretrained: bool) -> nn.Module:
        net: Optional[nn.Module] = None
        if pretrained:
            try:
                net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception as exc:  # offline / download failure
                logger.warning("Pretrained weights unavailable (%s); using random init.", exc)
        if net is None:
            net = models.resnet18(weights=None)
        net.fc = nn.Identity()  # expose the 512-d penultimate layer
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        return net.to(self.device)

    @torch.inference_mode()
    def extract(self, image_paths: Sequence[str]) -> np.ndarray:
        """Return an (n_images, 512) float32 array of latent vectors."""
        if not image_paths:
            raise DataValidationError("No images supplied.")
        feats: List[np.ndarray] = []
        bs = self.cfg.batch_size
        for start in range(0, len(image_paths), bs):
            batch = []
            for path in image_paths[start : start + bs]:
                try:
                    with Image.open(path) as img:
                        batch.append(self.transform(img.convert("RGB")))
                except Exception as exc:
                    raise DataValidationError(f"Cannot read image '{path}': {exc}") from exc
            tensor = torch.stack(batch).to(self.device)
            feats.append(self.model(tensor).cpu().numpy())
        return np.concatenate(feats).astype(np.float32)


# --------------------------------------------------------------------------- #
# Data preparation (split first, then fit preprocessing on train only)
# --------------------------------------------------------------------------- #
class DataPreparer:
    """Leakage-safe splitting and preprocessing."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def prepare(self, X: np.ndarray, y: Sequence[Any], modality: str) -> SplitData:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise DataValidationError("X must be 2-D and aligned with y.")
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise DataValidationError(f"Binary target required, found {len(classes)} classes.")
        if counts.min() < self.cfg.min_class_samples:
            raise DataValidationError(
                f"Minority class has {counts.min()} samples; need >= {self.cfg.min_class_samples}."
            )
        minority_label = classes[int(np.argmin(counts))]
        y_bin = (y == minority_label).astype(int)

        # 1) Stratified split BEFORE any fitting.
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y_bin, test_size=self.cfg.test_size, stratify=y_bin, random_state=self.cfg.random_state
        )
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_tmp, y_tmp, test_size=self.cfg.val_size, stratify=y_tmp, random_state=self.cfg.random_state
        )
        # 2) Imputer + scaler fitted on the training-fit partition ONLY.
        prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        X_fit_s = prep.fit_transform(X_fit)
        return SplitData(
            X_fit=X_fit_s, y_fit=y_fit,
            X_val=prep.transform(X_val), y_val=y_val,
            X_test=prep.transform(X_test), y_test=y_test,
            preprocessor=prep, minority_label=minority_label, modality=modality,
        )


def profile_dataset(data: SplitData, cfg: PipelineConfig) -> DatasetProfile:
    """Compute imbalance, dimensionality and noise on the training-fit partition."""
    X, y = data.X_fit, data.y_fit
    n_min, n_maj = int((y == 1).sum()), int((y == 0).sum())
    rng = np.random.default_rng(cfg.random_state)
    idx = np.arange(len(y))
    if len(idx) > cfg.noise_sample_cap:
        idx = rng.choice(idx, cfg.noise_sample_cap, replace=False)
    Xs, ys = X[idx], y[idx]
    k = min(cfg.noise_neighbors, len(ys) - 1)
    noise = 0.0
    min_idx = np.where(ys == 1)[0]
    if k >= 1 and len(min_idx) > 1:
        nn_ = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
        _, ind = nn_.kneighbors(Xs[min_idx])
        noise = float(np.mean(ys[ind[:, 1:]].sum(axis=1) == 0))
    return DatasetProfile(
        modality=data.modality, n_samples=len(y), n_features=X.shape[1],
        n_minority=n_min, n_majority=n_maj, minority_fraction=n_min / len(y),
        imbalance_ratio=n_maj / max(n_min, 1), noise_level=noise,
    )


# --------------------------------------------------------------------------- #
# Resampling suite
# --------------------------------------------------------------------------- #
def build_resampler(name: str, k_neighbors: int, sampling_ratio: float, n_minority: int, seed: int):
    """Create an imblearn sampler. k is clamped to what the minority class supports."""
    if name not in SUPPORTED_STRATEGIES:
        raise ResamplingError(f"Unknown strategy '{name}'. Choose from {SUPPORTED_STRATEGIES}.")
    k = max(1, min(int(k_neighbors), n_minority - 1))
    if name == "smote":
        return SMOTE(sampling_strategy=sampling_ratio, k_neighbors=k, random_state=seed)
    if name == "adasyn":
        return ADASYN(sampling_strategy=sampling_ratio, n_neighbors=k, random_state=seed)
    if name == "borderline_smote":
        return BorderlineSMOTE(
            sampling_strategy=sampling_ratio, k_neighbors=k, m_neighbors=max(10, k),
            kind="borderline-1", random_state=seed,
        )
    if name == "smote_tomek":
        inner = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=k, random_state=seed)
        return SMOTETomek(smote=inner, random_state=seed)
    return RandomOverSampler(sampling_strategy=sampling_ratio, random_state=seed)


def resample_train(X: np.ndarray, y: np.ndarray, name: str, k: int, ratio: float, seed: int):
    """Resample the TRAIN partition. If already at/above target ratio, returns it unchanged."""
    n_min, n_maj = int((y == 1).sum()), int((y == 0).sum())
    if ratio <= n_min / n_maj:
        logger.info("Target ratio %.3f <= current %.3f; skipping resampling.", ratio, n_min / n_maj)
        return X, y
    try:
        return build_resampler(name, k, ratio, n_min, seed).fit_resample(X, y)
    except ResamplingError:
        raise
    except Exception as exc:
        raise ResamplingError(f"{name} failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# RAG strategy engine (Qdrant)
# --------------------------------------------------------------------------- #
KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {"source": "Chawla et al. 2002 - SMOTE", "kind": "paper",
     "text": "standard smote moderate imbalance clean low noise interpolate minority neighbours k 5 balanced ratio",
     "rec": {"k_neighbors": 5, "sampling_ratio": 0.5}},
    {"source": "Han et al. 2005 - Borderline-SMOTE", "kind": "paper",
     "text": "extreme severe imbalance minority below two percent borderline samples near decision boundary small k",
     "rec": {"k_neighbors": 3, "sampling_ratio": 0.3}},
    {"source": "He et al. 2008 - ADASYN", "kind": "paper",
     "text": "adaptive synthetic hard to learn minority moderate imbalance clean low noise density distribution",
     "rec": {"k_neighbors": 5, "sampling_ratio": 0.5}},
    {"source": "Batista et al. 2004 - SMOTE-Tomek", "kind": "paper",
     "text": "noisy overlapping classes high noise clean tomek links after oversampling cleaning",
     "rec": {"k_neighbors": 5, "sampling_ratio": 0.5}},
    {"source": "Guideline - high-dimensional embeddings", "kind": "paper",
     "text": "high-dimensional embedding cnn latent vectors image features distances degrade larger k conservative ratio",
     "rec": {"k_neighbors": 7, "sampling_ratio": 0.3}},
    {"source": "Saito and Rehmsmeier 2015 - PR vs ROC", "kind": "evaluation",
     "text": "precision recall curve imbalanced evaluation f1 threshold optimisation pr-auc instead of accuracy",
     "rec": {"threshold_hint": 0.5}},
    {"source": "Sample policy - synthetic data scope", "kind": "compliance",
     "text": "compliance synthetic samples only in training never validation or test cap oversampling ratio",
     "rec": {"max_sampling_ratio": 0.5}},
    {"source": "Sample policy - reporting", "kind": "compliance",
     "text": "compliance report precision recall f1 pr-auc never accuracy document threshold source",
     "rec": {}},
]


class HashingEmbedder:
    """Deterministic, dependency-free text embedder (signed feature hashing)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        toks = re.findall(r"[a-z0-9\-]+", text.lower())
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        vec = np.zeros(self.dim, dtype=np.float32)
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        return (vec / norm if norm else vec).tolist()


@dataclass
class RetrievedStrategy:
    k_neighbors: int = 5
    sampling_ratio: float = 0.5
    threshold_hint: float = 0.5
    citations: List[str] = field(default_factory=list)
    backend: str = "defaults"


class RAGStrategyEngine:
    """Retrieves hyperparameter guidance from Qdrant; degrades gracefully if unavailable."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.embedder = HashingEmbedder(cfg.embedding_dim)
        self.client, self.backend = self._connect()
        try:
            self._seed()
        except Exception as exc:
            logger.warning("Could not seed knowledge base (%s); RAG falls back to defaults.", exc)
            self.client = None

    def _connect(self) -> Tuple[Optional[QdrantClient], str]:
        try:
            client = QdrantClient(url=self.cfg.qdrant_url, timeout=3)
            client.get_collections()
            return client, "qdrant"
        except Exception as exc:
            logger.warning("Qdrant unreachable at %s (%s); using in-memory store.", self.cfg.qdrant_url, exc)
        try:
            return QdrantClient(":memory:"), "qdrant-memory"
        except Exception as exc:
            logger.error("In-memory Qdrant failed: %s", exc)
            return None, "defaults"

    def _seed(self) -> None:
        if self.client is None:
            return
        name = self.cfg.qdrant_collection
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name, vectors_config=qm.VectorParams(size=self.cfg.embedding_dim, distance=qm.Distance.COSINE)
            )
        points = [
            qm.PointStruct(id=i, vector=self.embedder.embed(d["text"]), payload=d)
            for i, d in enumerate(KNOWLEDGE_BASE)
        ]
        self.client.upsert(name, points=points)

    @staticmethod
    def _query_text(p: DatasetProfile) -> str:
        sev = "extreme severe imbalance minority below two percent" if p.minority_fraction < 0.02 else "moderate imbalance"
        noise = "noisy overlapping classes high noise" if p.noise_level > 0.35 else "clean low noise"
        dim = "high-dimensional embedding cnn latent vectors" if p.n_features > 100 else "tabular"
        return f"{sev} {noise} {dim} smote compliance precision recall f1 pr-auc"

    def search(self, text: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Free-text lookup used by the assistant to ground its answers."""
        if self.client is None:
            return []
        try:
            hits = self.client.query_points(
                self.cfg.qdrant_collection, query=self.embedder.embed(text), limit=limit
            ).points
        except Exception as exc:
            logger.warning("RAG search failed (%s).", exc)
            return []
        return [{"source": (h.payload or {}).get("source", ""), "text": (h.payload or {}).get("text", ""),
                 "rec": (h.payload or {}).get("rec", {})} for h in hits if h.score > 0.05]

    def retrieve(self, profile: DatasetProfile) -> RetrievedStrategy:
        out = RetrievedStrategy(backend=self.backend)
        if self.client is None:
            return out
        try:
            hits = self.client.query_points(
                self.cfg.qdrant_collection, query=self.embedder.embed(self._query_text(profile)), limit=5
            ).points
        except Exception as exc:
            logger.warning("RAG lookup failed (%s); using defaults.", exc)
            out.backend = "defaults"
            return out
        got_paper = False
        for h in hits:
            doc = h.payload or {}
            rec, kind = doc.get("rec", {}), doc.get("kind")
            out.citations.append(f"{doc.get('source')} (score {h.score:.2f})")
            if kind == "paper" and not got_paper and "k_neighbors" in rec:
                out.k_neighbors, out.sampling_ratio, got_paper = rec["k_neighbors"], rec["sampling_ratio"], True
            elif kind == "evaluation":
                out.threshold_hint = rec.get("threshold_hint", out.threshold_hint)
            elif kind == "compliance" and "max_sampling_ratio" in rec:
                out.sampling_ratio = min(out.sampling_ratio, rec["max_sampling_ratio"])
        return out


# --------------------------------------------------------------------------- #
# Agentic orchestrator (LangGraph-style state machine)
# --------------------------------------------------------------------------- #
@dataclass
class AgentState:
    """Shared state passed between graph nodes."""

    data: SplitData
    override_strategy: Optional[str] = None
    override_k: Optional[int] = None
    override_ratio: Optional[float] = None
    classifier_name: str = "logreg"
    profile: Optional[DatasetProfile] = None
    strategy: str = "smote"
    k_neighbors: int = 5
    sampling_ratio: float = 0.5
    threshold_hint: float = 0.5
    citations: List[str] = field(default_factory=list)
    rag_backend: str = "defaults"
    routing_reason: str = ""
    X_res: Optional[np.ndarray] = None
    y_res: Optional[np.ndarray] = None
    model: Any = None
    attempts: int = 0
    errors: List[str] = field(default_factory=list)
    healing: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)


class StateMachine:
    """Tiny graph runner: each node returns the name of the next node, or None to stop."""

    def __init__(self, nodes: Dict[str, Callable[[AgentState], Optional[str]]], entry: str, max_steps: int = 40):
        self.nodes, self.entry, self.max_steps = nodes, entry, max_steps

    def run(self, state: AgentState) -> AgentState:
        current: Optional[str] = self.entry
        for _ in range(self.max_steps):
            if current is None:
                return state
            state.trace.append(current)
            current = self.nodes[current](state)
        raise PipelineError("State machine exceeded max steps (possible loop).")


class AgenticOrchestrator:
    """Inspects the dataset, routes to a resampler, and self-heals on failure."""

    def __init__(self, cfg: PipelineConfig, rag: RAGStrategyEngine) -> None:
        self.cfg, self.rag = cfg, rag
        self.planner = StateMachine(
            {"profile": self._profile, "retrieve": self._retrieve, "route": self._route,
             "resample": self._resample, "heal_resample": self._heal_resample},
            entry="profile",
        )
        self.trainer = StateMachine(
            {"train": self._train, "heal_train": self._heal_train}, entry="train"
        )

    # ---- planning graph ----
    def _profile(self, s: AgentState) -> str:
        s.profile = profile_dataset(s.data, self.cfg)
        return "retrieve"

    def _retrieve(self, s: AgentState) -> str:
        r = self.rag.retrieve(s.profile)
        s.k_neighbors, s.sampling_ratio, s.threshold_hint = r.k_neighbors, r.sampling_ratio, r.threshold_hint
        s.citations, s.rag_backend = r.citations, r.backend
        return "route"

    def _route(self, s: AgentState) -> str:
        p = s.profile
        if p.minority_fraction < 0.02:
            s.strategy, s.routing_reason = "borderline_smote", "minority < 2%: focus on boundary samples"
        elif p.noise_level > 0.35:
            s.strategy, s.routing_reason = "smote_tomek", "high class overlap: oversample then clean Tomek links"
        elif p.minority_fraction <= 0.10 and p.n_features <= 50:
            s.strategy, s.routing_reason = "adasyn", "moderate imbalance, low noise, low dimension"
        else:
            s.strategy, s.routing_reason = "smote", "milder imbalance: standard SMOTE"
        if s.override_strategy:
            s.strategy, s.routing_reason = s.override_strategy, "strategy set by caller"
        if s.override_k:
            s.k_neighbors = s.override_k
        if s.override_ratio:
            s.sampling_ratio = s.override_ratio
        return "resample"

    def _resample(self, s: AgentState) -> Optional[str]:
        try:
            s.X_res, s.y_res = resample_train(
                s.data.X_fit, s.data.y_fit, s.strategy, s.k_neighbors, s.sampling_ratio, self.cfg.random_state
            )
            return None
        except ResamplingError as exc:
            s.errors.append(str(exc))
            return "heal_resample"

    def _heal_resample(self, s: AgentState) -> str:
        s.attempts += 1
        if s.attempts > self.cfg.max_retries:
            raise ResamplingError(f"Resampling failed after {self.cfg.max_retries} retries: {s.errors}")
        if s.attempts == 1:
            s.k_neighbors = max(1, s.k_neighbors // 2)
            s.healing.append(f"halved k_neighbors to {s.k_neighbors}")
        elif s.attempts == 2:
            s.strategy, s.k_neighbors = "smote", min(3, s.k_neighbors)
            s.healing.append("fell back to plain SMOTE")
        else:
            s.strategy = "random_over"
            s.healing.append("fell back to random oversampling")
        return "resample"

    # ---- training graph ----
    def _train(self, s: AgentState) -> Optional[str]:
        try:
            s.model = build_classifier(s.classifier_name, s.attempts, self.cfg.random_state)
            s.model.fit(s.X_res, s.y_res)
            return None
        except Exception as exc:
            s.errors.append(f"train: {exc}")
            return "heal_train"

    def _heal_train(self, s: AgentState) -> str:
        s.attempts += 1
        if s.attempts > self.cfg.max_retries:
            raise TrainingError(f"Training failed after {self.cfg.max_retries} retries: {s.errors}")
        s.healing.append(f"retry {s.attempts}: simpler classifier settings")
        s.classifier_name = "logreg"
        return "train"

    def plan_and_resample(self, s: AgentState) -> AgentState:
        s.attempts = 0
        return self.planner.run(s)

    def train(self, s: AgentState) -> AgentState:
        if s.X_res is None:
            raise StageOrderError("Call /resample before /train.")
        s.attempts = 0
        return self.trainer.run(s)


def build_classifier(name: str, variant: int, seed: int):
    """Return a classifier. `variant` > 0 selects progressively more conservative settings."""
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
    if name == "logreg":
        solver = "lbfgs" if variant < 2 else "liblinear"
        return LogisticRegression(max_iter=2000, solver=solver, random_state=seed)
    raise TrainingError(f"Unknown classifier '{name}'. Choose 'logreg' or 'rf'.")


# --------------------------------------------------------------------------- #
# Honest evaluation
# --------------------------------------------------------------------------- #
class HonestEvaluator:
    """Precision, Recall, F1 and PR-AUC. Accuracy is deliberately never computed."""

    @staticmethod
    def optimize_threshold(y_true: np.ndarray, scores: np.ndarray, default: float = 0.5) -> Tuple[float, float]:
        """Return (threshold, F1) maximising F1 along the precision-recall curve."""
        p, r, t = precision_recall_curve(y_true, scores)
        if len(t) == 0:
            return default, 0.0
        f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
        i = int(np.nanargmax(f1))
        return float(t[i]), float(f1[i])

    @staticmethod
    def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
        pred = (scores >= threshold).astype(int)
        p, r, _ = precision_recall_curve(y_true, scores)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        return {
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true, scores)),
            "pr_auc_trapezoid": float(auc(r, p)),
            "pr_auc_baseline": float(np.mean(y_true)),  # a random model scores the prevalence
            "threshold": float(threshold),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }


# --------------------------------------------------------------------------- #
# Facade used by the API
# --------------------------------------------------------------------------- #
class SyntheticDataPipeline:
    """One dataset's lifecycle: ingest -> resample -> train -> evaluate."""

    def __init__(self, cfg: PipelineConfig, rag: RAGStrategyEngine,
                 extractor: Optional[CNNFeatureExtractor] = None) -> None:
        self.cfg, self.rag, self._extractor = cfg, rag, extractor
        self.orchestrator = AgenticOrchestrator(cfg, rag)
        self.evaluator = HonestEvaluator()
        self.data: Optional[SplitData] = None
        self.state: Optional[AgentState] = None
        self.last_metrics: Optional[Dict[str, Any]] = None

    # -- ingest --
    def ingest_tabular(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        if target not in df.columns:
            raise DataValidationError(f"Target column '{target}' not found.")
        df = df.dropna(subset=[target])
        X = pd.get_dummies(df.drop(columns=[target]), dummy_na=False).astype(float)
        if X.shape[1] == 0:
            raise DataValidationError("No feature columns found.")
        return self._finish_ingest(X.to_numpy(), df[target].to_numpy(), "tabular")

    def ingest_images(self, paths: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
        if self._extractor is None:
            self._extractor = CNNFeatureExtractor(self.cfg)
        feats = self._extractor.extract(paths)
        return self._finish_ingest(feats, np.asarray(labels), "image")

    def _finish_ingest(self, X: np.ndarray, y: np.ndarray, modality: str) -> Dict[str, Any]:
        self.data = DataPreparer(self.cfg).prepare(X, y, modality)
        self.state = AgentState(data=self.data, classifier_name=self.cfg.classifier)
        profile = profile_dataset(self.data, self.cfg)
        return {
            "profile": profile.to_dict(),
            "minority_label": str(self.data.minority_label),
            "split": {"fit": len(self.data.y_fit), "val": len(self.data.y_val), "test": len(self.data.y_test)},
        }

    def _require_data(self) -> AgentState:
        if self.state is None:
            raise StageOrderError("Call /ingest first.")
        return self.state

    # -- stages --
    def resample(self, strategy: str = "auto", k_neighbors: Optional[int] = None,
                 sampling_ratio: Optional[float] = None) -> Dict[str, Any]:
        s = self._require_data()
        s.override_strategy = None if strategy == "auto" else strategy
        s.override_k, s.override_ratio = k_neighbors, sampling_ratio
        s.errors, s.healing, s.trace, s.model = [], [], [], None
        before = {"minority": int((s.data.y_fit == 1).sum()), "majority": int((s.data.y_fit == 0).sum())}
        self.orchestrator.plan_and_resample(s)
        return {
            "profile": s.profile.to_dict(), "strategy": s.strategy, "routing_reason": s.routing_reason,
            "k_neighbors": s.k_neighbors, "sampling_ratio": s.sampling_ratio,
            "rag_backend": s.rag_backend, "rag_citations": s.citations,
            "counts_before": before,
            "counts_after": {"minority": int((s.y_res == 1).sum()), "majority": int((s.y_res == 0).sum())},
            "self_healing": s.healing, "errors": s.errors, "graph_trace": s.trace,
            "note": "Only the training-fit partition was resampled; validation and test are untouched.",
        }

    def train(self, classifier: Optional[str] = None) -> Dict[str, Any]:
        s = self._require_data()
        s.classifier_name = classifier or self.cfg.classifier
        s.errors, s.healing, s.trace = [], [], []
        self.orchestrator.train(s)
        return {"classifier": s.classifier_name, "train_rows": int(len(s.y_res)),
                "self_healing": s.healing, "errors": s.errors, "graph_trace": s.trace}

    def evaluate(self, threshold_source: Optional[str] = None) -> Dict[str, Any]:
        s = self._require_data()
        if s.model is None:
            raise StageOrderError("Call /train before /evaluate.")
        source = threshold_source or self.cfg.threshold_source
        val_scores = s.model.predict_proba(s.data.X_val)[:, 1]
        test_scores = s.model.predict_proba(s.data.X_test)[:, 1]
        if source == "test":
            thr, _ = self.evaluator.optimize_threshold(s.data.y_test, test_scores, s.threshold_hint)
        else:
            thr, _ = self.evaluator.optimize_threshold(s.data.y_val, val_scores, s.threshold_hint)
        metrics = self.evaluator.evaluate(s.data.y_test, test_scores, thr)
        metrics.update({
            "threshold_source": source,
            "optimistic": source == "test",
            "test_counts": {"minority": int((s.data.y_test == 1).sum()), "majority": int((s.data.y_test == 0).sum())},
            "note": "Accuracy intentionally omitted. Test set was never resampled.",
        })
        self.last_metrics = metrics
        return metrics

    def compare_strategies(self, classifier: str = "logreg") -> Dict[str, Any]:
        """Benchmark every resampler on the VALIDATION partition (test is never touched)."""
        s = self._require_data()
        d, rows = s.data, []
        k, ratio = s.k_neighbors, s.sampling_ratio
        baseline = build_classifier(classifier, 0, self.cfg.random_state).fit(d.X_fit, d.y_fit)
        rows.append(self._score_row("no_resampling", baseline, d))
        for name in SUPPORTED_STRATEGIES:
            try:
                Xr, yr = resample_train(d.X_fit, d.y_fit, name, k, ratio, self.cfg.random_state)
                model = build_classifier(classifier, 0, self.cfg.random_state).fit(Xr, yr)
                rows.append(self._score_row(name, model, d))
            except Exception as exc:  # one failing strategy must not break the benchmark
                rows.append({"strategy": name, "error": str(exc)})
        ranked = sorted([r for r in rows if "pr_auc_val" in r], key=lambda r: r["pr_auc_val"], reverse=True)
        return {"metric": "pr_auc_val", "best": ranked[0]["strategy"] if ranked else None, "results": rows,
                "note": "Scored on validation data only; the test set is untouched."}

    def _score_row(self, name: str, model: Any, d: SplitData) -> Dict[str, Any]:
        sc = model.predict_proba(d.X_val)[:, 1]
        thr, _ = self.evaluator.optimize_threshold(d.y_val, sc)
        m = self.evaluator.evaluate(d.y_val, sc, thr)
        return {"strategy": name, "pr_auc_val": m["pr_auc"], "precision_val": m["precision"],
                "recall_val": m["recall"], "f1_val": m["f1"]}

    def context(self) -> Dict[str, Any]:
        """Compact, non-sensitive summary of this session for the assistant."""
        s = self.state
        if s is None:
            return {}
        ctx: Dict[str, Any] = {"modality": s.data.modality}
        if s.profile:
            ctx.update({"profile": s.profile.to_dict(), "strategy": s.strategy,
                        "routing_reason": s.routing_reason, "k_neighbors": s.k_neighbors,
                        "sampling_ratio": s.sampling_ratio, "classifier": s.classifier_name})
        if self.last_metrics:
            keep = ("precision", "recall", "f1", "pr_auc", "pr_auc_baseline", "threshold", "threshold_source")
            ctx["test_metrics"] = {k: self.last_metrics[k] for k in keep}
        return ctx

    def report(self) -> Dict[str, Any]:
        """Model card: what was done, why, and the honest numbers."""
        s = self._require_data()
        if self.last_metrics is None:
            raise StageOrderError("Call /evaluate before requesting a report.")
        m = self.last_metrics
        md = [
            "# Model card",
            f"- Modality: {s.data.modality}; minority fraction {s.profile.minority_fraction:.2%}; noise {s.profile.noise_level:.2f}",
            f"- Strategy: {s.strategy} ({s.routing_reason}); k={s.k_neighbors}; target ratio={s.sampling_ratio}",
            f"- Classifier: {s.classifier_name}",
            f"- Test PR-AUC {m['pr_auc']:.3f} (random baseline {m['pr_auc_baseline']:.3f}); "
            f"precision {m['precision']:.3f}; recall {m['recall']:.3f}; F1 {m['f1']:.3f}",
            f"- Threshold {m['threshold']:.3f} tuned on {m['threshold_source']}" + (" (optimistic)" if m["optimistic"] else ""),
            "- Leakage controls: split before scaling; scaler fit on train only; resampling on train only.",
            "- Accuracy is intentionally not reported.",
        ]
        return {"markdown": "\n".join(md), "metrics": m, "citations": s.citations, "self_healing": s.healing}


def make_demo_dataset(n: int = 4000, minority_fraction: float = 0.01, n_features: int = 8,
                      seed: int = 42) -> pd.DataFrame:
    """Synthetic imbalanced tabular data with overlapping classes, for demos and tests."""
    rng = np.random.default_rng(seed)
    n_min = max(int(n * minority_fraction), 12)
    n_maj = n - n_min
    maj = rng.normal(0.0, 1.0, size=(n_maj, n_features))
    mino = rng.normal(1.4, 1.2, size=(n_min, n_features))
    X = np.vstack([maj, mino])
    y = np.r_[np.zeros(n_maj, dtype=int), np.ones(n_min, dtype=int)]
    idx = rng.permutation(n)
    df = pd.DataFrame(X[idx], columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y[idx]
    return df
