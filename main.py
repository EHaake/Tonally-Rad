import argparse
import cv2
import numpy as np


is_drawing = False
prev_x, prev_y = None, None


def draw(event, x, y, flags, param):
    global prev_x, prev_y, is_drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        prev_x, prev_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            cv2.line(img, (prev_x, prev_y), (x, y),
                     color=(0, 0, 255), thickness=5)
            prev_x, prev_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        cv2.line(img, (prev_x, prev_y), (x, y), color=(0, 0, 255), thickness=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drawing')
    parser.add_argument("image_path", help="Path to image")
    args = parser.parse_args()
    image_path = args.image_path

    img = cv2.imread(image_path)

    cv2.namedWindow('draw')
    cv2.setMouseCallback('draw', draw)

    while True:
        cv2.imshow('draw', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
