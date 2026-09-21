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
from collections import Counter
from datetime import datetime

from app import face_db
from app.face_training import MIN_SAMPLES_PER_CLASS, train_classifier

# Below this (but still >= MIN_SAMPLES_PER_CLASS, so still trainable), an
# employee's samples are flagged as thin rather than excluded — purely
# informational, per the "flag, don't auto-exclude" instruction.
LOW_SAMPLE_WARNING_THRESHOLD = 10


def main() -> int:
    face_db.init_face_tables()  # safe no-op if already created by the running backend

    by_status = face_db.get_capture_counts_by_status()
    labeled_rows = face_db.get_labeled_training_embeddings()
    per_employee = Counter(r["person_id"] for r in labeled_rows)

    print(f"Training Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"Labeled samples available: {len(labeled_rows)}")
    print(f"Employees/classes represented: {len(per_employee)}")
    if per_employee:
        print("Samples per employee:")
        for emp_id, n in sorted(per_employee.items()):
            if n < MIN_SAMPLES_PER_CLASS:
                flag = "  <- below minimum, excluded from this run"
            elif n < LOW_SAMPLE_WARNING_THRESHOLD:
                flag = "  <- few samples, consider labeling more"
            else:
                flag = ""
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
        print("The previously trained model (if any) was left untouched and is still in use.")
        return 1
    except Exception as e:
        print(f"\nTraining FAILED (unexpected error): {e}")
        print("The previously trained model (if any) was left untouched and is still in use.")
        return 1

    print("Training: fit LogisticRegression on frozen ArcFace embeddings")
    print(f"  trained on {result['trained_on_samples']} samples across {len(result['employees'])} employees")
    if result["excluded_too_few_samples"]:
        print(f"  excluded (too few samples, need >= {MIN_SAMPLES_PER_CLASS}): {result['excluded_too_few_samples']}")

    print("-" * 60)
    if result["validation"]:
        v = result["validation"]
        print(
            f"Validation accuracy: {v['accuracy'] * 100:.1f}% on {v['val_samples']} held-out samples "
            f"across {v['val_classes']} employees (grouped by tracked appearance, not seen in training)"
        )
        print(
            "  This is a HELD-OUT VALIDATION measurement, not a live-camera accuracy figure — "
            "see the /face-training model history panel for that distinction."
        )
        pcv = result.get("per_class_validation")
        if pcv:
            print("  Per-employee (held-out): accuracy, false-rejections, false-acceptances")
            for emp_id, m in sorted(pcv["per_class"].items()):
                flag = ""
                if m["insufficient_samples"]:
                    flag = "  <- too few held-out samples for a reliable per-class number"
                elif m["weak"]:
                    flag = "  <- WEAK: below the 70% per-class floor"
                print(
                    f"    {emp_id}: {m['accuracy']*100:.0f}% ({m['correct']}/{m['val_samples']}), "
                    f"false_rej={m['false_rejections']} false_acc={m['false_acceptances']}{flag}"
                )
    else:
        print(f"Validation: {result['validation_skipped_reason']}")

    print("-" * 60)
    print(f"Model saved to: {result['model_path']}")
    if result["backup_path"]:
        print(f"Previous model backed up to: {result['backup_path']}")

    history = face_db.list_training_runs(limit=5)
    if len(history) > 1:
        print("-" * 60)
        print("Recent training history:")
        for i, run in enumerate(history):
            acc = f"{run['validation_accuracy'] * 100:.1f}%" if run["validation_accuracy"] is not None else "n/a"
            when = datetime.fromtimestamp(run["trained_at"]).strftime("%Y-%m-%d %H:%M")
            marker = " (this run)" if i == 0 else ""
            print(f"  {when}  {run['sample_count']} samples, {run['class_count']} classes, {acc} val accuracy{marker}")

    print("=" * 60)
    print("Training completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
