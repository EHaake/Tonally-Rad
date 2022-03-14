import argparse
import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

# Constants alpha, lambda, and epsilon
ALPHA = 1
LAMBDA = 0.2
EPSILON = 0.0001


def select_pixels(e, x, y, flags, param):
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
                     color=(0, 255, 0), thickness=8)
            prev_x, prev_y = x, y
            selected_points.append([x, y])

    elif e == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        constrain_pixels(selected_points)
        selected_points = []  # reset selected points


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
    # weights = np.ones(mask.shape[0], mask.shape[1])
    # print(weights.shape)

    if gaussian:
        l_weight = np.exp(-((l_chan - l_mean) ** 2) /
                          (intensity_threshold ** 2))
        # a_weight = np.exp(-((a_chan - a_mean) ** 2) /
        #                   (chroma_threshold ** 2))
        # b_weight = np.exp(-((b_chan - b_mean) ** 2) /
        #                   (chroma_threshold ** 2))
        # weights = np.array([l_weight, a_weight, b_weight])
        weights = np.array([l_weight])
    else:
        weights = np.array([1., 1., 1.])

    # divide every component of weight by number of elements
    # weights /= weights.size
    # weights /= 3

    # create the output image showing the selected pixels
    overlay_image = cv2.merge((l_chan, a_chan, b_chan))
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_LAB2BGR)

    # set the selected pixels to magenta
    overlay_image[mask] = (255, 0, 255)

    # show the image
    cv2.imshow('masked_img', overlay_image)


def lischinski_tmo():
    """
    Apply the Lischinski TMO to the image
    """
    # split the lab image into its channels
    l_chan, a_chan, b_chan = cv2.split(lab_img)

    # extract the max and min of the l channel
    min_l_log = np.log2(np.min(l_chan) + EPSILON)
    if min_l_log < 0:
        min_l_log = EPSILON
    max_l_log = np.log2(np.max(l_chan))
    print("min_l_log: {}, max_l_log: {}".format(min_l_log, max_l_log))

    num_zones = int(np.ceil(max_l_log - min_l_log))

    # choose the representative Rz (luminance value) for each zone z
    f_stop_map = np.zeros(l_chan.shape)
    # log_avg = np.log2(np.mean(l_chan))
    log_avg = log_mean(l_chan)
    print("log_avg: {}".format(log_avg))

    for i in range(num_zones):
        lower_l = np.power(2, i - 1 + min_l_log)
        upper_l = np.power(2, min_l_log + i)
        indices = np.where(np.logical_and(
            l_chan >= lower_l, l_chan < upper_l))
        # print("indices: {}".format(indices))

        if len(indices) > 0:
            Rz = np.median(l_chan[indices])
            Rz = (ALPHA * Rz) / log_avg
            f = Rz / (1 + Rz)
            # apply np.log2(f / Rz) to f_stop_map at indices
            # print("indices: ", indices)
            f_stop_map[indices] = np.log2(f / Rz)
            # if i is 1:
            # print("f_stop_map shape: {}".format(f_stop_map.shape))

    weights = np.ones(l_chan.shape) * 0.07

    # fill f_stop_map with random values between 0 and 1
    # f_stop_map = np.random.rand(l_chan.shape[0], l_chan.shape[1])

    # compute the expsoure function
    exposure_factor = compute_exposure_fn(
        np.log(EPSILON + l_chan), f_stop_map, weights, log_avg)
    # exposure_factor = compute_exposure_fn(
    #     l_chan, f_stop_map, weights, log_avg)
    # exposure_factor = np.ones(l_chan.shape)

    # apply the exposure factor to the image
    l_chan = np.uint8(l_chan * exposure_factor)
    output_image = cv2.merge((l_chan, a_chan, b_chan))
    output_image = cv2.cvtColor(output_image, cv2.COLOR_LAB2BGR)

    # correct the gamma of the tone mapped image
    # output_image = gamma_correction(output_image, 2.0)

    # show the image
    # cv2.imshow('output_image', output_image)
    print("==========================")
    print("DONE :)")
    print("==========================")
    return output_image


def compute_exposure_fn(l_chan, f_stop_map, weights, log_avg):
    """
    Compute the exposure_fn by solving Lischinski minimization function
    Solve the linear system A f = b, where b = w * g
    :param l_chan: the l channel of the image
    :param f_stop_map: the f_stop map
    :param weights: the weights of the pixels
    """
    h, w = f_stop_map.shape
    n = w * h

    # generate the b vector
    g = log_avg * weights
    print("g max: {}, g min: {}".format(np.max(g), np.min(g)))
    # reshape g into a vector b
    b = np.reshape(g, (n, 1))

    x_gradient, y_gradient = compute_gradients(l_chan)

    # generate the A matrix
    A = generate_A(x_gradient, y_gradient, h, w, f_stop_map)

    # Solve the sparse linear system using Conjugate Gradients
    f, info = spl.cg(A, b, tol=1e-6, maxiter=100)
    print('info: {}'.format(info))
    f = np.reshape(f, (h, w), order='F')
    print("Max value in f: {}".format(np.max(f)))

    return f


def generate_A(x_gradient, y_gradient, h, w, f_stop_map):
    """
    Generate the A matrix
    :param x_gradient: the x gradient
    :param y_gradient: the y gradient
    :param h: the height of the image
    :param w: the width of the image
    :return: the A matrix
    """
    n = h * w
    A = sp.diags([x_gradient, y_gradient], offsets=[-h, -1],
                 shape=(n, n), dtype=np.float32)
    A = A + A.T

    g00 = np.pad(x_gradient, 0, 'constant')
    # g00 = g00[0: -h]

    g01 = np.pad(y_gradient, 0, 'constant')
    # g01 = g01[0: -1]
    # D = np.reshape(weights, (n, 1)) - (g00 + x_gradient + g01 + y_gradient)
    D = np.reshape(f_stop_map, (n, 1)) - (g00 + x_gradient + g01 + y_gradient)
    print("D shape: {}".format(D.shape))
    # A = A + np.diag(D)
    # A = A + sp.diags(D, 0, (n, n), dtype=np.float32)
    A = A + D

    return A


def compute_gradients(l_chan):
    """
    Compute the gradients of the image
    :param l_chan: the l channel of the image
    :return: the gradients of the image
    """
    y_gradient = np.diff(l_chan, 1, axis=0)
    x_gradient = np.diff(l_chan, 1, axis=1)

    y_gradient = (-LAMBDA) / (np.power(np.abs(y_gradient), ALPHA) + EPSILON)
    x_gradient = (-LAMBDA) / (np.power(np.abs(x_gradient), ALPHA) + EPSILON)

    y_gradient = np.pad(y_gradient, ((0, 1), (0, 0)), 'constant')
    x_gradient = np.pad(x_gradient, ((0, 0), (0, 1)), 'constant')

    # reshape into a column vector
    y_gradient = np.reshape(
        y_gradient, y_gradient.shape[0] * y_gradient.shape[1], order='C')
    x_gradient = np.reshape(
        x_gradient, x_gradient.shape[0] * x_gradient.shape[1], order='C')

    return x_gradient, y_gradient


def gamma_correction(img, gamma=1.0):
    """
    Apply gamma correction to the image
    :param img: the image
    :param gamma: the gamma value
    :return: the gamma corrected image
    """
    inverse_gamma = 1.0 / gamma
    values = np.array([((i / 255.0) ** inverse_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, values)


def log_mean(lum, delta=0.000001):
    """
    Compute the log mean of the luminance
    :param lum: the luminance channel
    :param delta: the delta
    :return: the log mean
    """
    # return np.log(np.mean(np.exp(lum))) / np.log(1 + delta)
    return np.exp(np.mean(np.log(lum + delta)))


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
    # cv2.namedWindow('draw')
    # cv2.setMouseCallback('draw', draw)

    auto_img = lischinski_tmo()
    # keep the window open until the user presses escape key
    while True:
        # cv2.imshow('draw', img)
        cv2.imshow('auto', auto_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
