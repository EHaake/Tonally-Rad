import argparse
import cv2
import numpy as np


def draw(e, x, y, flags, param):
    '''
    Draws a line from the previous point to the current point using the mouse
    Indicate the areas of the image that should be selected
    :param e: event
    :param x: x mouse coordinate
    :param y: y mouse coordinate
    '''
    global prev_x, prev_y, is_drawing, selected_points

    if e == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        prev_x, prev_y = x, y

    elif e == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            cv2.line(img, (prev_x, prev_y), (x, y),
                     color=(0, 255, 0), thickness=10)
            prev_x, prev_y = x, y
            selected_points.append([x, y])

    elif e == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        constrain_pixels(selected_points)
        selected_points = []


def constrain_pixels(points, intensity_threshold=20):
    """
    Find the pixels in the image that are within the constraint.
    A pixel with a lightness value of l is only selected if abs(mean - l) < intensity_threshold
    :param points: list of (x, y) tuples
    :param intensity_threshold: threshold for the intensity value
    """
    # remove duplicate points
    points = np.unique(np.array(points), axis=0)

    # split the lab image into its channels
    l_chan, a_chan, b_chan = cv2.split(lab_img)

    # collect the channel values of the selected pixels
    x_pts = points[:, 0]
    y_pts = points[:, 1]
    l_vals, a_vals, b_vals = l_chan[y_pts,
                                    x_pts], a_chan[y_pts, x_pts], b_chan[y_pts, x_pts]

    # find the mean of the channel values
    l_mean, a_mean, b_mean = np.mean(l_vals), np.mean(a_vals), np.mean(b_vals)
    print(l_mean, a_mean, b_mean)


if __name__ == '__main__':
    # parse args or print usage
    parser = argparse.ArgumentParser(description='Drawing')
    parser.add_argument("image_path", help="Path to image")
    args = parser.parse_args()
    image_path = args.image_path

    # load the image using opencv
    img = cv2.imread(image_path)
    # convert image to LAB space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # declare our global variables
    is_drawing = False
    prev_x, prev_y = None, None
    selected_points = []

    # create a window and bind the mouse callback function
    cv2.namedWindow('draw')
    cv2.setMouseCallback('draw', draw)

    # keep the window open until the user presses escape key
    while True:
        cv2.imshow('draw', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
