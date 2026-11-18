import io
import time

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="VisionPort Gateway - Session 3 Request Trace",
    description="YOLOv8 inference pipeline tracing API",
    version="1.0.0"
)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

yolo = YOLO("../yolov8n.pt")

net = yolo.model.to(device).eval()

names = yolo.names

IMGSZ = 640


# --------------------------------------------------
# GPU SYNCHRONIZATION
# --------------------------------------------------

def sync():
    """Wait for GPU operations to finish for accurate timing."""

    if device == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------
# HOME ROUTE
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "VisionPort Gateway is running successfully!",
        "status": "success",
        "device": device,
        "model": "YOLOv8n"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": "YOLOv8n"
    }


# --------------------------------------------------
# YOLOV8 INFERENCE TRACE
# --------------------------------------------------

@app.post("/trace")
def trace(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45
):

    # Validate image format
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail="Send a JPEG or PNG image"
        )

    if not 0 <= conf <= 1:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1"
        )

    if not 0 <= iou <= 1:
        raise HTTPException(
            status_code=400,
            detail="IoU must be between 0 and 1"
        )

    stages = []

    # Record execution stages
    def add(name, where, fmt, shape, t0, note=""):

        stages.append({
            "stage": len(stages) + 1,
            "name": name,
            "where": where,
            "format": fmt,
            "shape": shape,
            "note": note,
            "ms": round(
                (time.perf_counter() - t0) * 1000, 2
            )
        })

    t_all = time.perf_counter()

    # --------------------------------------------------
    # STAGE 1: RECEIVE UPLOAD
    # --------------------------------------------------

    t0 = time.perf_counter()

    data = file.file.read()

    add(
        "receive upload",
        "CPU / gateway",
        "bytes",
        f"{len(data)} bytes",
        t0
    )

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    # --------------------------------------------------
    # STAGE 2: DECODE IMAGE
    # --------------------------------------------------

    t0 = time.perf_counter()

    try:
        image = Image.open(
            io.BytesIO(data)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image"
        )

    arr = np.array(image)

    orig_hw = arr.shape[:2]

    add(
        "decode image",
        "CPU / gateway",
        "numpy uint8 (H,W,C)",
        list(arr.shape),
        t0
    )

    # --------------------------------------------------
    # STAGE 3: LETTERBOX RESIZE
    # --------------------------------------------------

    t0 = time.perf_counter()

    boxed = LetterBox(
        new_shape=(IMGSZ, IMGSZ),
        auto=False
    )(image=arr)

    add(
        "letterbox resize",
        "CPU / gateway",
        "numpy uint8 (H,W,C)",
        list(boxed.shape),
        t0,
        "Aspect ratio preserved, grey padding added"
    )

    # --------------------------------------------------
    # STAGE 4: CONVERT TO TENSOR
    # --------------------------------------------------

    t0 = time.perf_counter()

    x = (
        torch.from_numpy(boxed)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .float() / 255.0
    )

    add(
        "to tensor",
        "CPU / gateway",
        "torch float32 (N,C,H,W)",
        list(x.shape),
        t0,
        "Channels first, batch dimension added, scaled to 0-1"
    )

    # --------------------------------------------------
    # STAGE 5: HOST TO DEVICE TRANSFER
    # --------------------------------------------------

    t0 = time.perf_counter()

    x = x.to(device)

    sync()

    add(
        "host to device",
        f"CPU -> {device}",
        "torch float32 (N,C,H,W)",
        list(x.shape),
        t0,
        "Copy over PCIe" if device == "cuda"
        else "No copy required; already on CPU"
    )

    # --------------------------------------------------
    # STAGE 6: MODEL FORWARD PASS
    # --------------------------------------------------

    t0 = time.perf_counter()

    with torch.no_grad():
        out = net(x)

    sync()

    raw = out[0] if isinstance(out, (list, tuple)) else out

    add(
        "model forward",
        device,
        "torch float32 (N,4+classes,anchors)",
        list(raw.shape),
        t0,
        "YOLOv8 raw prediction output"
    )

    # --------------------------------------------------
    # STAGE 7: NON-MAXIMUM SUPPRESSION
    # --------------------------------------------------

    t0 = time.perf_counter()

    dets = non_max_suppression(
        raw,
        conf,
        iou
    )[0]

    sync()

    add(
        "NMS",
        device,
        "torch float32 (n,6)",
        list(dets.shape),
        t0,
        "Columns: x1, y1, x2, y2, confidence, class id"
    )

    # --------------------------------------------------
    # STAGE 8: RESCALE BOUNDING BOXES
    # --------------------------------------------------

    t0 = time.perf_counter()

    dets[:, :4] = ops.scale_boxes(
        x.shape[2:],
        dets[:, :4],
        orig_hw
    )

    add(
        "rescale boxes",
        device,
        "torch float32 (n,6)",
        list(dets.shape),
        t0,
        "Boxes mapped to original image coordinates"
    )

    # --------------------------------------------------
    # STAGE 9: SERIALIZE RESPONSE
    # --------------------------------------------------

    t0 = time.perf_counter()

    rows = dets.cpu().tolist()

    detections = [
        {
            "class": names[int(c)],
            "confidence": round(s, 3),
            "box_xyxy": [
                round(v, 1)
                for v in (x1, y1, x2, y2)
            ]
        }
        for x1, y1, x2, y2, s, c in rows
    ]

    add(
        "serialise response",
        "CPU / gateway",
        "JSON",
        f"{len(detections)} detections",
        t0
    )

    # --------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------

    sync()

    total_ms = round(
        (time.perf_counter() - t_all) * 1000,
        2
    )

    return {
        "project": "VisionPort",
        "device": device,
        "model": "YOLOv8n",
        "original_size_hw": list(orig_hw),
        "image_size": IMGSZ,
        "confidence_threshold": conf,
        "iou_threshold": iou,
        "stages": stages,
        "total_ms": total_ms,
        "detections_count": len(detections),
        "detections": detections
    }