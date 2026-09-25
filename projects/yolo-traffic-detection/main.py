"""Real-time object detection on a live video stream with YOLO11.

    python main.py --source "https://www.youtube.com/watch?v=<live-id>"

Press ``q`` in the video window to quit.
"""

import argparse

import cv2

# BGR colours (OpenCV convention) for the classes worth highlighting.
LABEL_COLORS = {
    "car": (255, 0, 0),
    "person": (0, 255, 0),
    "bicycle": (0, 0, 255),
    "bus": (255, 255, 0),
    "truck": (0, 255, 255),
    "traffic light": (255, 0, 255),
}
DEFAULT_COLOR = (255, 255, 255)


def annotate(frame, model, conf_threshold=0.5):
    """Draw a box and a label for every detection above the confidence threshold.

    Args:
        frame: BGR image, modified in place.
        model: an Ultralytics YOLO model (anything callable that returns results).
        conf_threshold: minimum confidence to keep a detection.

    Returns:
        The annotated frame.
    """
    for box in model(frame, verbose=False)[0].boxes:
        confidence = float(box.conf[0])
        if confidence <= conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return frame


def main():
    from ultralytics import YOLO
    from vidgear.gears import CamGear

    parser = argparse.ArgumentParser(description="YOLO11 detection on a live stream.")
    parser.add_argument(
        "--source",
        default="https://www.youtube.com/watch?v=1EiC9bvVGnk",
        help="YouTube live URL (the default stream may be offline)",
    )
    parser.add_argument("--model", default="yolo11s.pt", help="YOLO weights")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    args = parser.parse_args()

    model = YOLO(args.model)
    stream = CamGear(
        source=args.source, stream_mode=True, logging=True, STREAM_RESOLUTION="720p"
    ).start()
    try:
        while (frame := stream.read()) is not None:
            cv2.imshow("YOLO11 detection", annotate(frame, model, args.conf))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        stream.stop()


if __name__ == "__main__":
    main()
