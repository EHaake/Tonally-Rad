from operator import is_
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
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('draw')
    cv2.setMouseCallback('draw', draw)

    while True:
        cv2.imshow('draw', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
