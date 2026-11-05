import numpy as np
import pytest
from fastapi.testclient import TestClient

import pipeline as pl
from pipeline import PipelineConfig, RAGStrategyEngine, SyntheticDataPipeline, make_demo_dataset


@pytest.fixture(scope="module")
def rag():
    return RAGStrategyEngine(PipelineConfig(qdrant_url="http://127.0.0.1:1"))


def _pipe(rag, frac=0.01):
    p = SyntheticDataPipeline(PipelineConfig(qdrant_url="http://127.0.0.1:1"), rag)
    p.ingest_tabular(make_demo_dataset(4000, frac), "target")
    return p


def test_routes_borderline_when_minority_under_2pct(rag):
    assert _pipe(rag, 0.01).resample()["routing_reason"].startswith("minority < 2%")


def test_only_train_partition_is_resampled(rag):
    p = _pipe(rag)
    n_test, n_val = len(p.data.y_test), len(p.data.y_val)
    X_test_before = p.data.X_test.copy()
    out = p.resample()
    assert out["counts_after"]["minority"] > out["counts_before"]["minority"]
    assert len(p.data.y_test) == n_test and len(p.data.y_val) == n_val
    assert np.array_equal(p.data.X_test, X_test_before)


def test_scaler_fitted_on_train_only(rag):
    p = _pipe(rag)
    assert np.allclose(p.data.X_fit.mean(axis=0), 0, atol=1e-8)  # exact only for the fit partition


def test_metrics_have_no_accuracy(rag):
    p = _pipe(rag, 0.05)
    p.resample(); p.train()
    m = p.evaluate()
    assert "accuracy" not in m and {"precision", "recall", "f1", "pr_auc"} <= set(m)
    assert m["optimistic"] is False


def test_self_healing_recovers(rag, monkeypatch):
    real, calls = pl.resample_train, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise pl.ResamplingError("boom")
        return real(*a, **k)

    monkeypatch.setattr(pl, "resample_train", flaky)
    out = _pipe(rag).resample()
    assert len(out["self_healing"]) == 2 and out["counts_after"]["minority"] > 0


def test_threshold_optimizer_finds_perfect_split():
    thr, f1 = pl.HonestEvaluator.optimize_threshold(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
    assert f1 == pytest.approx(1.0) and 0.2 < thr <= 0.8


def test_api_end_to_end():
    from main import app
    with TestClient(app) as c:
        csv = c.get("/demo-data", params={"n": 3000, "minority_fraction": 0.02}).content
        r = c.post("/ingest", files={"file": ("d.csv", csv)}, data={"kind": "tabular", "target": "target"})
        assert r.status_code == 200, r.text
        did = r.json()["dataset_id"]
        assert c.post("/train", json={"dataset_id": did}).status_code == 409  # stage order enforced
        assert c.post("/resample", json={"dataset_id": did}).status_code == 200
        assert c.post("/train", json={"dataset_id": did}).status_code == 200
        assert c.post("/compare", json={"dataset_id": did}).json()["best"]
        ev = c.post("/evaluate", json={"dataset_id": did}).json()
        assert "accuracy" not in ev
        assert "Model card" in c.get(f"/report/{did}").json()["markdown"]
