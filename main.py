import argparse
import cv2
import numpy as np
import pcg


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


def constrain_pixels(points, intensity_threshold=20, chroma_threshold=10, gaussian=True):
    """
    Find the pixels in the image that are within the constraint.
    A pixel with a lightness value of l is only selected if abs(mean - l) < intensity_threshold
    and the euclidian distance between a and b channels is < chroma_threshold
    :param points: list of (x, y) tuples
    :param intensity_threshold: threshold for the intensity value
    :param chroma_threshold: threshold for the chroma value
    :param gaussian_falloff: whether to use a gaussian falloff
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

    # select pixels in the image that are within the constraint
    # mask = np.logical_and(np.abs(l_chan - l_mean) < intensity_threshold,
    #                       np.sqrt(((a_mean-a_chan) ** 2) + ((b_mean-b_chan) ** 2)) < chroma_threshold)
    mask = np.logical_and(np.abs(l_chan - l_mean) < intensity_threshold,
                          np.abs(a_chan - a_mean) < chroma_threshold,
                          np.abs(b_chan - b_mean) < chroma_threshold)

    # per pixel weight constraint mask
    weights = np.ones(mask.shape[0], mask.shape[1])
    print(weights.shape)

    if gaussian:
        l_weight = np.exp(-((l_chan - l_mean) ** 2) /
                          (intensity_threshold ** 2))
        a_weight = np.exp(-((a_chan - a_mean) ** 2) /
                          (chroma_threshold ** 2))
        b_weight = np.exp(-((b_chan - b_mean) ** 2) /
                          (chroma_threshold ** 2))
        weights = np.array([l_weight, a_weight, b_weight])
    else:
        weights = np.array([1., 1., 1.])

    # divide every component of weight by number of elements
    # weights /= weights.size
    # weights /= 3

    print(weights.shape)

    # create the output image showing the selected pixels
    output_image = cv2.merge((l_chan, a_chan, b_chan))
    output_image = cv2.cvtColor(output_image, cv2.COLOR_LAB2BGR)

    # set the selected pixels to magenta
    output_image[mask] = (255, 0, 255)

    # show the image
    cv2.imshow('masked_img', output_image)


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
