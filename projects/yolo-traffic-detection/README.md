# YOLO traffic detection

> Real-time object detection on a YouTube live stream: frames are pulled with VidGear, detected with a pretrained YOLO11 model and drawn with OpenCV.

<!-- ![demo](assets/demo.gif) -->

## Why

A minimal end-to-end inference loop on a live source: stream ingestion, model inference, visualisation. It is the starting point for the serving and optimisation projects listed in the [roadmap](../../ROADMAP.md).

## Approach

- **Input:** any YouTube live URL, read at 720p through `CamGear` (yt-dlp under the hood).
- **Model:** `yolo11s.pt`, pretrained on COCO. There is no fine-tuning here, so the project is about the streaming and inference loop rather than model quality.
- **Output:** boxes and labels for detections above a confidence threshold (0.5 by default). Cars, people, bicycles, buses, trucks and traffic lights get their own colour.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --source "https://www.youtube.com/watch?v=<live-id>"
```

Options: `--model` (any Ultralytics weights, downloaded on first use) and `--conf` (confidence threshold). Press `q` in the video window to quit. The default stream may be offline, in which case pass another live URL.

## Tests

```bash
pip install pytest numpy opencv-python-headless
python -m pytest
```

The tests check the drawing logic with a fake model, so they need neither weights nor a network connection.

## Notes

- Weights (`*.pt`) are git-ignored and downloaded automatically by Ultralytics.
- Next steps: measure FPS and latency, fine-tune on a custom traffic dataset, export to ONNX.
- Model: Ultralytics YOLO11. Dataset used for pretraining: COCO.
