"""Replay a recorded video through the full Staff Count pipeline and print every
event, for testing on real footage (low light, crowded entrance, ...).

Uses a throwaway database, so the live counts are never touched. Run from backend/:

    python -m scripts.replay_staff --video entrance.mp4 --line 0.2,0.5,0.8,0.5 --inside 1
    python -m scripts.replay_staff --video entrance.mp4 --line 0.2,0.5,0.8,0.5 --faces --out annotated.mp4

--line    entry line as x1,y1,x2,y2 fractions of the frame
--inside  1 or -1: which side of the line is the office (flip it if entries show as exits)
--roi     optional office area polygon as x,y,x,y,... fractions
--faces   also run face recognition (slower; needs the trained classifier)
--out     write an annotated video (boxes, track ids, names, the line, the count)

Record a test clip from a camera with ffmpeg, e.g. 60 seconds:
    ffmpeg -rtsp_transport tcp -i "rtsp://user:pass@host:port/path" -t 60 -c copy entrance.ts
"""

import argparse
import tempfile
import time
from pathlib import Path

import cv2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--line", required=True)
    ap.add_argument("--inside", type=int, default=1)
    ap.add_argument("--roi")
    ap.add_argument("--faces", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    from app.staff import config as staff_config
    from app.staff import occupancy
    from app.staff.occupancy import StaffOccupancyManager
    from app.staff.service import MODEL_PATH, _Identity
    from app.staff.tracker import StaffCameraTracker
    from ultralytics import YOLO

    cfg = staff_config.SETTINGS
    occupancy.DB_PATH = Path(tempfile.mkdtemp()) / "replay.db"
    occupancy.init_db()
    manager = StaffOccupancyManager(cfg)
    x1, y1, x2, y2 = (float(v) for v in args.line.split(","))
    roi = None
    if args.roi:
        v = [float(x) for x in args.roi.split(",")]
        roi = [[v[i], v[i + 1]] for i in range(0, len(v), 2)]
    tracker = StaffCameraTracker(1, manager, [[x1, y1], [x2, y2]], args.inside, roi, cfg)
    identity = _Identity()
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(fps / cfg["process_fps"]))
    writer = None
    t0 = time.time()
    n = 0
    print(f"{args.video}: {fps:.0f} fps, analysing every {step} frame(s) (~{fps / step:.1f} fps)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % step:
            continue
        ts = t0 + n / fps  # video time, so cooldowns/timeouts behave as they would live
        res = model.track(frame, persist=True, tracker=cfg["tracker"], classes=[0],
                          conf=cfg["detection_confidence"], imgsz=cfg["imgsz"], verbose=False)[0]
        people = []
        if res.boxes is not None and res.boxes.id is not None:
            people = [{"track_id": int(t), "bbox": [float(v) for v in b]}
                      for t, b in zip(res.boxes.id.cpu().numpy().astype(int), res.boxes.xyxy.cpu().numpy())]
        for ev in tracker.update(people, frame.shape, ts,
                                 (lambda b, frame=frame: identity.identify(frame, b)) if args.faces else None,
                                 lambda b, frame=frame: identity.embed(frame, b)):
            print(f"  {n / fps:7.1f}s  {ev['event_type']:<16} track {ev['track_id']:<4} "
                  f"{ev['employee_id'] or 'unknown':<8} ({ev['identity_source'] or ''}) {ev['details'] or ''}")
        if args.out:
            writer = writer or cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (frame.shape[1], frame.shape[0]))
            h, w = frame.shape[:2]
            cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 255), 3)
            for t in tracker.debug(ts):
                bx = [int(v) for v in t["bbox"]]
                color = (0, 200, 0) if t["side"] == "in" else (0, 0, 230) if t["side"] == "out" else (200, 200, 200)
                cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
                cv2.putText(frame, f"#{t['track_id']} {t['employee_id'] or ''} {t['side'] or ''}", (bx[0], bx[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            c = manager.counts(anonymous_mode=not args.faces)
            cv2.putText(frame, f"Inside: {c['total_persons']}  In: {c['total_entries_today']}  Out: {c['total_exits_today']}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            writer.write(frame)
    if writer:
        writer.release()
    c = manager.counts(anonymous_mode=not args.faces)
    print(f"\ninside at the end: {c['total_persons']} (known {c['known_staff']}, unknown {c['unknown_persons']}) | "
          f"entries {c['total_entries_today']} | exits {c['total_exits_today']} | "
          f"duplicates prevented {c['duplicates_prevented_today']} | unmatched exits {c['unmatched_exits_today']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
