"""Model provisioning (offline mode), classifier compatibility, config validation."""

import json
import os
import time

import numpy as np
import pytest

from app import classifier_io, config, models


def _trained_classifier():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    y = np.array(["001"] * 20 + ["002"] * 20)
    X[20:] += 3
    return LogisticRegression(max_iter=200).fit(X, y)


class BrokenClassifier:
    """Loads fine, then fails on predict — like a 1.9.1 pickle under 1.5.2."""
    classes_ = np.array(["001"])
    n_features_in_ = 8

    def predict_proba(self, X):
        raise AttributeError("'LogisticRegression' object has no attribute 'multi_class'")


def test_save_writes_metadata_and_loads_verified(tmp_path):
    path = tmp_path / "classifier.joblib"
    classifier_io.save(_trained_classifier(), path)
    meta = json.loads((tmp_path / "classifier.meta.json").read_text())
    import sklearn

    assert meta["sklearn"] == sklearn.__version__ and meta["n_features"] == 8 and meta["classes"] == ["001", "002"]
    clf = classifier_io.load_verified(path)
    assert clf is not None and classifier_io.last_status["loaded"]


def test_an_unusable_classifier_is_refused_not_crashed_on(tmp_path):
    import joblib

    path = tmp_path / "classifier.joblib"
    joblib.dump(BrokenClassifier(), path)
    assert classifier_io.load_verified(path) is None
    assert classifier_io.last_status["loaded"] is False
    assert "multi_class" in classifier_io.last_status["reason"]


def test_missing_classifier_is_reported(tmp_path):
    assert classifier_io.load_verified(tmp_path / "nope.joblib") is None
    assert classifier_io.last_status["reason"] == "no trained classifier"


def test_backups_are_rotated_and_the_active_model_is_kept(tmp_path):
    path = tmp_path / "classifier.joblib"
    classifier_io.save(_trained_classifier(), path)
    now = time.time()
    for i in range(8):
        b = tmp_path / f"classifier.joblib.bak-{1000 + i}"
        b.write_bytes(b"old model")
        os.utime(b, (now - 100 + i, now - 100 + i))
    removed = classifier_io.rotate_backups(path, keep=3)
    kept = sorted(p.name for p in tmp_path.glob("classifier.joblib.bak-*"))
    assert kept == ["classifier.joblib.bak-1005", "classifier.joblib.bak-1006", "classifier.joblib.bak-1007"]
    assert len(removed) == 5 and path.exists()


def test_saving_twice_quickly_keeps_both_backups(tmp_path):
    path = tmp_path / "classifier.joblib"
    for _ in range(3):
        classifier_io.save(_trained_classifier(), path)
        time.sleep(0.002)
    assert len(list(tmp_path.glob("classifier.joblib.bak-*"))) == 2


def test_offline_mode_refuses_missing_models_with_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_OFFLINE_MODE", True)
    with pytest.raises(models.ModelMissingError) as e:
        models.require("yolo_person")
    assert "fetch_models" in str(e.value) and "yolov8n.pt" in str(e.value)
    assert "yolo_person" in models.missing_required()


def test_online_mode_lets_the_library_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_OFFLINE_MODE", False)
    assert models.require("yolo_person") == tmp_path / "yolov8n.pt"


def test_present_models_are_found(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_OFFLINE_MODE", True)
    (tmp_path / "yolov8n.pt").write_bytes(b"weights")
    assert models.require("yolo_person") == tmp_path / "yolov8n.pt"


def test_offline_environment_disables_hub_downloads(monkeypatch):
    monkeypatch.setattr(config, "MODEL_OFFLINE_MODE", True)
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    models.apply_offline_environment()
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_reid_embedder_refuses_imagenet_fallback_offline(monkeypatch):
    from app.reid import reid_embedding

    monkeypatch.setattr(config, "MODEL_OFFLINE_MODE", True)
    with pytest.raises(models.ModelMissingError):
        reid_embedding.BodyReIdEmbedder("osnet_x0_25", "", "cpu")


@pytest.fixture
def prod(monkeypatch):
    monkeypatch.setattr(config, "IS_PRODUCTION", True)
    monkeypatch.setattr(config, "APP_ENV", "production")
    monkeypatch.setattr(config, "JWT_SECRET_IS_EPHEMERAL", False)
    monkeypatch.setattr(config, "JWT_SECRET", "x" * 64)
    monkeypatch.setattr(config, "MODEL_OFFLINE_MODE", True)
    monkeypatch.setattr(config, "IDENTITY_SERVICE_BASE", "https://identity.example.com")
    monkeypatch.setattr(config, "CORS_ORIGINS", ["https://vision.example.com"])
    return monkeypatch


def test_a_good_production_config_validates(prod):
    assert config.validate() == []


@pytest.mark.parametrize("attr,value,needle", [
    ("JWT_SECRET_IS_EPHEMERAL", True, "JWT_SECRET"),
    ("JWT_SECRET", "short", "32 characters"),
    ("MODEL_OFFLINE_MODE", False, "MODEL_OFFLINE_MODE"),
    ("IDENTITY_SERVICE_BASE", "http://13.61.58.14", "https"),
    ("CORS_ORIGINS", ["*"], "never '*'"),
    ("CORS_ORIGINS", ["http://vision.example.com"], "https"),
])
def test_unsafe_production_configs_are_rejected(prod, attr, value, needle):
    prod.setattr(config, attr, value)
    problems = config.validate()
    assert problems and any(needle in p for p in problems)


def test_production_startup_refuses_an_unsafe_config(prod):
    from app import main

    prod.setattr(config, "JWT_SECRET_IS_EPHEMERAL", True)
    with pytest.raises(RuntimeError, match="unsafe production configuration"):
        main.check_configuration()


def test_production_startup_refuses_without_an_admin(prod, isolated_db):
    from app import main

    prod.setattr(main, "_enable_wal", lambda: None)
    with pytest.raises(RuntimeError, match="no admin account"):
        main.init_databases()


def test_bad_numeric_env_values_are_clear_errors(monkeypatch):
    monkeypatch.setenv("RTSP_READ_TIMEOUT", "ten")
    with pytest.raises(ValueError, match="RTSP_READ_TIMEOUT"):
        config._float("RTSP_READ_TIMEOUT", 1.0)
