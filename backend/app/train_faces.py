"""
train_faces.py

The one terminal command for retraining the employee classifier, with no
browser/backend-running dependency:

    cd backend
    python -m app.train_faces

Does exactly what POST /api/faces/training/train does (see
face_training.train_classifier docstring) — this just wraps it with a
plain-terminal report and a real process exit code, per the FACE_TRAINING.md
"train from the terminal" requirement. Uses the same face_training_captures
rows already sitting in backend/data/app.db; it does not re-run YOLO/ArcFace
over the stored JPEGs, since valid embeddings are already stored per row.

Never runs on its own — no scheduler, no import-time side effect, nothing in
main.py calls this. Only runs when a person types the command above.
"""

import sys

from app import face_db
from app.face_training import MIN_SAMPLES_PER_CLASS, train_classifier


def main() -> int:
    face_db.init_face_tables()  # safe no-op if already created by the running backend

    by_status = face_db.get_capture_counts_by_status()
    labeled_rows = face_db.get_labeled_training_embeddings()
    from collections import Counter

    per_employee = Counter(r["person_id"] for r in labeled_rows)

    print("Face Classifier Training")
    print("=" * 60)
    print(f"Labeled samples available: {len(labeled_rows)}")
    print(f"Employees/classes represented: {len(per_employee)}")
    if per_employee:
        print("Samples per employee:")
        for emp_id, n in sorted(per_employee.items()):
            flag = "  <- below minimum, excluded" if n < MIN_SAMPLES_PER_CLASS else ""
            print(f"    {emp_id}: {n}{flag}")
    print(
        f"Excluded (never trainable): {by_status.get('no_embedding', 0)} no_embedding, "
        f"{by_status.get('rejected', 0)} rejected"
    )
    print("-" * 60)

    try:
        result = train_classifier()
    except ValueError as e:
        print(f"\nTraining FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nTraining FAILED (unexpected error): {e}")
        return 1

    print("Training: fit LogisticRegression on frozen ArcFace embeddings")
    print(f"  trained on {result['trained_on_samples']} samples across {len(result['employees'])} employees")
    if result["excluded_too_few_samples"]:
        print(f"  excluded (too few samples, need >= {MIN_SAMPLES_PER_CLASS}): {result['excluded_too_few_samples']}")

    print("-" * 60)
    if result["validation"]:
        v = result["validation"]
        print(
            f"Validation: {v['accuracy'] * 100:.1f}% accuracy on {v['val_samples']} held-out samples "
            f"across {v['val_classes']} employees (grouped by tracked appearance, not seen in training)"
        )
    else:
        print(f"Validation: {result['validation_skipped_reason']}")

    print("-" * 60)
    print(f"Model saved to: {result['model_path']}")
    if result["backup_path"]:
        print(f"Previous model backed up to: {result['backup_path']}")
    print("=" * 60)
    print("Training completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
