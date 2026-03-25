#!/usr/bin/env python3
"""
detect_persons.py — Detecta personas en el video original con YOLOv8
y dibuja los bounding boxes sobre el video overlay producido por SAM2.

Uso:
    python detect_persons.py \
        --original  ruta/al/video_original.mp4 \
        --overlay   outputs/20250315_142301/overlay.mp4 \
        --output    outputs/20250315_142301/overlay_persons.mp4 \
        --stride    2 \
        --conf      0.4

    # stride debe coincidir con el que usaste en la app SAM2
    # Si no recuerdas el stride, revisa session_points.json (campo "stride")

Instalar dependencia si no está:
    pip install ultralytics
"""

import argparse, sys, cv2
import numpy as np
from pathlib import Path

# ── YOLO ──────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics no instalado. Corre:\n   pip install ultralytics")
    sys.exit(1)

PERSON_CLASS = 0          # COCO class 0 = person
BOX_COLOR    = (0, 165, 255)   # BGR rojo suave
BOX_THICK    = 2

# ─────────────────────────────────────────────────────────────────────────────

def detect_persons(
    original_path: str,
    overlay_path:  str,
    output_path:   str,
    stride:        int,
    conf:          float,
    model_path:    str,
):
    original_path = Path(original_path)
    overlay_path  = Path(overlay_path)
    output_path   = Path(output_path)

    if not original_path.exists():
        print(f"❌ Video original no encontrado: {original_path}")
        sys.exit(1)
    if not overlay_path.exists():
        print(f"❌ Video overlay no encontrado: {overlay_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    print(f"► Cargando YOLOv8 desde '{model_path}' (se descarga si no existe)...")
    model = YOLO(model_path)
    print(f"  ✅ Modelo cargado")

    # ── Leer dimensiones del video original ───────────────────────────────────
    cap_orig = cv2.VideoCapture(str(original_path))
    orig_w   = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h   = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_n   = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap_orig.get(cv2.CAP_PROP_FPS) or 30.0

    # ── Leer dimensiones del overlay ──────────────────────────────────────────
    cap_over = cv2.VideoCapture(str(overlay_path))
    over_w   = int(cap_over.get(cv2.CAP_PROP_FRAME_WIDTH))
    over_h   = int(cap_over.get(cv2.CAP_PROP_FRAME_HEIGHT))
    over_n   = int(cap_over.get(cv2.CAP_PROP_FRAME_COUNT))
    over_fps = cap_over.get(cv2.CAP_PROP_FPS) or (orig_fps / stride)
    cap_orig.release()
    cap_over.release()

    # ── Factores de escala coordenadas original → overlay ─────────────────────
    scale_x = over_w / orig_w
    scale_y = over_h / orig_h

    print(f"► Original : {orig_w}×{orig_h} · {orig_n} frames · {orig_fps:.1f} FPS")
    print(f"► Overlay  : {over_w}×{over_h} · {over_n} frames · {over_fps:.1f} FPS")
    print(f"► Stride   : {stride}  →  se usa 1 de cada {stride} frames del original")
    print(f"► Escala   : x={scale_x:.3f}  y={scale_y:.3f}")
    print(f"► Confianza mínima: {conf}")
    print()

    # Advertencia si los números no cuadran
    expected_over_n = orig_n // stride
    if abs(over_n - expected_over_n) > 2:
        print(f"⚠️  Frames esperados en overlay: ~{expected_over_n}, reales: {over_n}")
        print(f"   Verifica que --stride coincide con el usado en la app SAM2")
        print()

    # ── Detección en el video original (solo frames que corresponden al overlay) ──
    print("► Detectando personas en el video original...")
    # detections[overlay_frame_idx] = list of (x1,y1,x2,y2) en coords overlay
    detections: dict[int, list] = {}

    cap_orig = cv2.VideoCapture(str(original_path))
    raw_idx  = 0
    while True:
        ret, frame = cap_orig.read()
        if not ret:
            break

        # Solo procesar frames que tienen correspondencia en el overlay
        if raw_idx % stride == 0:
            overlay_idx = raw_idx // stride

            results = model(
                frame,
                classes=[PERSON_CLASS],
                conf=conf,
                verbose=False,
            )

            boxes_in_overlay = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Escalar coordenadas de resolución original a resolución overlay
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                boxes_in_overlay.append((x1, y1, x2, y2))

            detections[overlay_idx] = boxes_in_overlay

        raw_idx += 1

    cap_orig.release()
    total_persons = sum(len(v) for v in detections.values())
    frames_with_persons = sum(1 for v in detections.values() if v)
    print(f"  ✅ {total_persons} detecciones en {frames_with_persons}/{over_n} frames del overlay")
    print()

    # ── Escribir video de salida ───────────────────────────────────────────────
    print(f"► Escribiendo video con bounding boxes → {output_path}")
    cap_over = cv2.VideoCapture(str(overlay_path))
    writer   = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        over_fps,
        (over_w, over_h),
    )

    over_idx = 0
    while True:
        ret, frame = cap_over.read()
        if not ret:
            break

        for (x1, y1, x2, y2) in detections.get(over_idx, []):
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)

        writer.write(frame)
        over_idx += 1

    cap_over.release()
    writer.release()

    print(f"  ✅ Video guardado: {output_path}")
    print(f"     {over_idx} frames escritos")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import json

    ap = argparse.ArgumentParser(
        description="Dibuja personas detectadas por YOLO sobre el overlay de SAM2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--original", required=True,
                    help="Ruta al video original de drone")
    ap.add_argument("--overlay",  required=True,
                    help="Ruta al overlay.mp4 producido por la app SAM2")
    ap.add_argument("--output",   default=None,
                    help="Ruta del video de salida (default: overlay_persons.mp4 junto al overlay)")
    ap.add_argument("--stride",   type=int, default=None,
                    help="Stride usado en la app SAM2 (default: lee session_points.json)")
    ap.add_argument("--conf",     type=float, default=0.4,
                    help="Confianza mínima YOLO para personas")
    ap.add_argument("--model",    default="yolov8n.pt",
                    help="Modelo YOLO a usar (se descarga si no existe)")
    args = ap.parse_args()

    # Resolver stride desde session_points.json si no se pasó
    stride = args.stride
    if stride is None:
        session = Path("session_points.json")
        if session.exists():
            try:
                data   = json.loads(session.read_text())
                stride = int(data.get("stride", 1))
                print(f"► Stride leído de session_points.json: {stride}")
            except Exception:
                stride = 1
        else:
            stride = 1
            print("⚠️  --stride no especificado y session_points.json no encontrado. Usando stride=1")

    # Resolver output path
    output = args.output
    if output is None:
        overlay_p = Path(args.overlay)
        output    = str(overlay_p.parent / "overlay_persons.mp4")

    detect_persons(
        original_path = args.original,
        overlay_path  = args.overlay,
        output_path   = output,
        stride        = stride,
        conf          = args.conf,
        model_path    = args.model,
    )


if __name__ == "__main__":
    main()

