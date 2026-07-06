import os
import cv2
import numpy as np

def yolo_seg_to_mask(label_path, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()
        if len(values) < 7:
            continue

        cls_id = int(float(values[0]))
        if cls_id != 0:
            continue

        coords = list(map(float, values[1:]))
        points = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * width)
            y = int(coords[i + 1] * height)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)

    return mask