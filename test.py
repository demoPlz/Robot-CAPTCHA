#!/usr/bin/env python3
import glob, os, re, sys, cv2

# Heuristics to ignore RealSense nodes
REALSENSE_TERMS = ("realsense", "depth", "infrared", "stereo module", "motion module")

def _read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def _video_nodes():
    devs = sorted(glob.glob("/dev/video*"), key=lambda p: int(re.search(r"\d+$", p).group()))
    out = []
    for dev in devs:
        idx = int(re.search(r"\d+$", dev).group())
        name = _read(f"/sys/class/video4linux/video{idx}/name")
        devlink = os.path.realpath(f"/sys/class/video4linux/video{idx}/device")
        m = re.search(r"/usb\d+/(.+?):\d+\.\d+$", devlink)
        if not m:
            m = re.search(r"/usb\d+/(.+)$", devlink)
        group_id = m.group(1) if m else f"idx-{idx}"
        out.append({"idx": idx, "dev": dev, "name": name, "group": group_id.lower()})
    return out

def _is_realsense(name: str) -> bool:
    n = (name or "").lower()
    return any(t in n for t in REALSENSE_TERMS)

def _works_with(cap: cv2.VideoCapture) -> bool:
    ok, _ = cap.read()
    return ok

def _pick_best_index(indices):
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if _works_with(cap):
            cap.release()
            return idx
        cap.release()
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened() and _works_with(cap):
            cap.release()
            return idx
        cap.release()
    return None

def list_physical_webcams():
    nodes = _video_nodes()
    nodes = [n for n in nodes if not _is_realsense(n["name"])]
    groups = {}
    for n in nodes:
        groups.setdefault((n["group"], n["name"]), []).append(n["idx"])
    cameras = []
    for (group_id, name), idxs in groups.items():
        best = _pick_best_index(idxs)
        cameras.append({
            "group": group_id,
            "name": name or "Unknown",
            "indices": sorted(idxs),
            "best_index": best
        })
    cameras.sort(key=lambda c: (9999 if c["best_index"] is None else c["best_index"]))
    return cameras

if __name__ == "__main__":
    cams = list_physical_webcams()
    if not cams:
        print("No physical webcams found (non-RealSense).")
        sys.exit(0)

    print("Physical webcams (deduplicated):")
    for i, c in enumerate(cams):
        mark = "✓" if c["best_index"] is not None else "✗"
        print(f"  [{i}] {mark} {c['name']}  usb={c['group']}  nodes={c['indices']}  use_index={c['best_index']}")

    # Open all found webcams simultaneously
    caps = []
    for c in cams:
        if c["best_index"] is None:
            continue
        cap = cv2.VideoCapture(c["best_index"], cv2.CAP_V4L2)
        if cap.isOpened():
            caps.append((c, cap))

    if not caps:
        print("No usable webcams could be opened.")
        sys.exit(0)

    print("\nPress 'q' to quit all streams.")
    while True:
        for c, cap in caps:
            ok, frame = cap.read()
            if not ok:
                continue
            # Overlay the cv2 index
            idx = c["best_index"]
            cv2.putText(frame, f"Index {idx}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow(f"Cam {idx} - {c['name']}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for _, cap in caps:
        cap.release()
    cv2.destroyAllWindows()
