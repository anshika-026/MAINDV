"""Enrollment upload hardening (app/uploads.py, POST /api/faces/enroll)."""

import cv2
import numpy as np
import pytest

from app import config, face_db, uploads
from tests.conftest import bearer


def _jpeg(w=64, h=64) -> bytes:
    img = np.full((h, w, 3), 128, np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 3, (200, 180, 160), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _png() -> bytes:
    return cv2.imencode(".png", np.full((32, 32, 3), 90, np.uint8))[1].tobytes()


TRAVERSAL_IDS = [
    "../", "../../", "..\\..\\", "../../app/main", "..%2F..%2Fetc", "%2e%2e%2f",
    "/etc/passwd", "C:\\Windows\\win.ini", "C:", "\\\\server\\share", "abc\x00.jpg",
    "a/b", "a\\b", ".", "..", "", " 018", "018 ", "x" * 65, "emp id",
]


@pytest.mark.parametrize("bad", TRAVERSAL_IDS)
def test_identifier_validation_rejects_traversal_and_junk(bad):
    with pytest.raises(uploads.UploadError):
        uploads.validate_identifier(bad)


@pytest.mark.parametrize("ok", ["018", "EMP-2201", "sonam_k", "A", "x" * 64])
def test_identifier_validation_accepts_real_ids(ok):
    assert uploads.validate_identifier(ok) == ok


@pytest.mark.parametrize("name", ["../x.jpg", "..", "a/b.jpg", "a\\b.jpg", "x\x00.jpg", "", "."])
def test_safe_child_never_escapes(tmp_path, name):
    with pytest.raises(uploads.UploadError):
        uploads.safe_child(tmp_path, name)


def test_decode_accepts_real_images():
    assert uploads.decode_image(_jpeg()).shape == (64, 64, 3)
    assert uploads.decode_image(_png()).shape == (32, 32, 3)


@pytest.mark.parametrize("data,status", [
    (b"", 400),
    (b"%PDF-1.7 not an image", 415),
    (b"<?php system($_GET['c']); ?>", 415),
    (b"MZ\x90\x00 fake exe", 415),
    (b"\xff\xd8\xff" + b"\x00" * 100, 400),      # JPEG magic, garbage body
    (b"GIF89a....", 415),
])
def test_decode_rejects_non_images(data, status):
    with pytest.raises(uploads.UploadError) as e:
        uploads.decode_image(data)
    assert e.value.status_code == status


def test_decode_rejects_oversized():
    big = b"\xff\xd8\xff" + b"\x00" * (config.MAX_UPLOAD_BYTES + 10)
    with pytest.raises(uploads.UploadError) as e:
        uploads.decode_image(big)
    assert e.value.status_code == 413


def test_save_image_uses_server_names_and_never_overwrites(tmp_path):
    img = uploads.decode_image(_jpeg())
    a = uploads.save_image(img, tmp_path, "018")
    b = uploads.save_image(img, tmp_path, "018")
    assert a != b and a.parent == tmp_path.resolve() and b.parent == tmp_path.resolve()
    assert a.suffix == ".jpg" and a.name.startswith("018_")
    assert cv2.imread(str(a)) is not None


class _Face:
    bbox = (0, 0, 10, 10)
    normed_embedding = np.ones(512, np.float32) / np.sqrt(512)


@pytest.fixture
def enroll_env(api, admin_token, tmp_path, monkeypatch):
    from app import face_routes

    enroll_dir = tmp_path / "face_enroll"
    enroll_dir.mkdir()
    monkeypatch.setattr(face_routes, "ENROLL_DIR", str(enroll_dir))
    monkeypatch.setattr(face_routes, "_largest_face", lambda img: _Face())
    return api, bearer(admin_token), enroll_dir


def _files_under(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize("person_id", TRAVERSAL_IDS[:12])
def test_enroll_rejects_traversal_person_ids_and_writes_nothing(enroll_env, tmp_path, person_id):
    api, h, enroll_dir = enroll_env
    before = _files_under(tmp_path)
    r = api.post("/api/faces/enroll", headers=h, data={"person_id": person_id},
                 files={"photo": ("face.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code in (400, 422)
    assert _files_under(tmp_path) == before


@pytest.mark.parametrize("filename", ["../../evil.jpg", "shell.php", "x.jpg.exe", "C:\\evil.jpg", "a\x00.jpg"])
def test_enroll_ignores_the_client_filename(enroll_env, filename):
    api, h, enroll_dir = enroll_env
    r = api.post("/api/faces/enroll", headers=h, data={"person_id": "018"},
                 files={"photo": (filename, _jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    [saved] = _files_under(enroll_dir)
    assert saved.suffix == ".jpg" and saved.name.startswith("018_")


def test_enroll_rejects_fake_extension_non_image(enroll_env):
    api, h, enroll_dir = enroll_env
    r = api.post("/api/faces/enroll", headers=h, data={"person_id": "018"},
                 files={"photo": ("face.jpg", b"#!/bin/sh\nrm -rf /", "image/jpeg")})
    assert r.status_code == 415
    assert _files_under(enroll_dir) == []


def test_enroll_rejects_oversized_body(enroll_env):
    api, h, enroll_dir = enroll_env
    big = b"\xff\xd8\xff" + b"\x00" * (config.MAX_UPLOAD_BYTES + 512 * 1024)
    r = api.post("/api/faces/enroll", headers=h, data={"person_id": "018"},
                 files={"photo": ("face.jpg", big, "image/jpeg")})
    assert r.status_code == 413
    assert _files_under(enroll_dir) == []


def test_enroll_with_no_face_writes_nothing(enroll_env, monkeypatch):
    from app import face_routes

    api, h, enroll_dir = enroll_env
    monkeypatch.setattr(face_routes, "_largest_face", lambda img: None)
    r = api.post("/api/faces/enroll", headers=h, data={"person_id": "018"},
                 files={"photo": ("face.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 422
    assert _files_under(enroll_dir) == []


def test_duplicate_uploads_create_separate_files_and_relative_db_paths(enroll_env):
    api, h, enroll_dir = enroll_env
    for _ in range(2):
        assert api.post("/api/faces/enroll", headers=h, data={"person_id": "018"},
                        files={"photo": ("face.jpg", _jpeg(), "image/jpeg")}).status_code == 200
    assert len(_files_under(enroll_dir)) == 2
    stored = [e["source_image_path"] for e in face_db.list_person_embeddings("018")]
    assert all(p.startswith("face_enroll/") and "\\" not in p and ":" not in p for p in stored)


def test_enroll_requires_admin(api):
    r = api.post("/api/faces/enroll", data={"person_id": "018"}, files={"photo": ("f.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 401


@pytest.mark.parametrize("employee_id", ["../../x", "..\\x", "/abs", "a/b"])
def test_training_routes_reject_path_like_employee_ids(api, admin_token, employee_id):
    h = bearer(admin_token)
    assert api.post("/api/faces/training/employees", headers=h, json={"employee_id": employee_id, "name": "X"}).status_code == 422
    assert api.post("/api/faces/training/label", headers=h, json={"capture_id": 1, "employee_id": employee_id}).status_code == 422
