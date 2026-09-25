from types import SimpleNamespace

import numpy as np
from main import annotate


def box(xyxy, conf, cls):
    """Mimic an Ultralytics box: every field is indexable with [0]."""
    return SimpleNamespace(xyxy=[xyxy], conf=[conf], cls=[cls])


class FakeModel:
    names = {0: "person", 2: "car"}

    def __init__(self, boxes):
        self.boxes = boxes

    def __call__(self, frame, verbose=False):
        return [SimpleNamespace(boxes=self.boxes)]


def test_draws_confident_detections_and_skips_weak_ones():
    frame = np.zeros((100, 100, 3), np.uint8)
    model = FakeModel([box((10, 10, 50, 50), 0.9, 2), box((60, 60, 90, 90), 0.3, 0)])
    annotate(frame, model)
    assert frame[10, 10].tolist() == [255, 0, 0]  # car, blue in BGR
    assert frame[60, 60].tolist() == [0, 0, 0]  # below the 0.5 threshold


def test_persons_are_drawn_in_green():
    frame = np.zeros((100, 100, 3), np.uint8)
    annotate(frame, FakeModel([box((10, 10, 50, 50), 0.9, 0)]))
    assert frame[10, 10].tolist() == [0, 255, 0]
