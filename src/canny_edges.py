import numpy as np
import math


def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def convolution(image_bw, kernel):
    output = np.zeros(image_bw.shape)

    pad_height = int((kernel.shape[0] - 1) / 2)
    pad_width = int((kernel.shape[1] - 1) / 2)

    padded_image = np.zeros(
        (image_bw.shape[0] + (2 * pad_height), image_bw.shape[1] + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height,
                 pad_width:padded_image.shape[1] - pad_width] = image_bw

    for row in range(image_bw.shape[0]):
        for col in range(image_bw.shape[1]):
            output[row, col] = np.sum(
                kernel * padded_image[row:row + kernel.shape[0], col:col + kernel.shape[1]])
            output[row, col] /= kernel.shape[0] * kernel.shape[1]

    return output


def convert_to_grayscale(image):
    gray_image = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    return gray_image.astype(np.uint8)


def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    return convolution(convert_to_grayscale(image), kernel)


def sobel_edge_detection(blurred_img, filter):
    new_image_x = convolution(blurred_img, filter)
    new_image_y = convolution(blurred_img, np.flip(filter.T, axis=0))
    gradient_magnitude = np.sqrt(
        np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    gradient_direction = np.arctan2(new_image_y, new_image_x)
    gradient_direction = np.rad2deg(gradient_direction)
    gradient_direction += 180
    return gradient_magnitude, gradient_direction


def non_max_suppression(gradient_magnitude, gradient_direction):
    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)
    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]

    return output


def threshold(image, low, high, weak):
    output = np.zeros(image.shape)
    strong = 255
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    return output


def hysteresis(image, weak):
    image_row, image_col = image.shape
    top_to_bottom = image.copy()

    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                        row + 1, col] == 255 or top_to_bottom[
                        row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[row - 1, col + 1] == 255 or top_to_bottom[
                        row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0

    bottom_to_top = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                        row + 1, col] == 255 or bottom_to_top[
                        row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[row - 1, col + 1] == 255 or bottom_to_top[
                        row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0

    right_to_left = image.copy()

    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                        row + 1, col] == 255 or right_to_left[
                        row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[row - 1, col + 1] == 255 or right_to_left[
                        row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    left_to_right = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                        row + 1, col] == 255 or left_to_right[
                        row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[row - 1, col + 1] == 255 or left_to_right[
                        row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right
    final_image[final_image > 255] = 255
    return final_image


def detect_canny_edges(img_bw, kernel_size=7, low=5, high=20, weak=50):
    blurred_image = gaussian_blur(img_bw, kernel_size)
    edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_magnitude, gradient_direction = sobel_edge_detection(blurred_image, edge_filter)
    canny_edges = non_max_suppression(gradient_magnitude, gradient_direction)
    canny_edges = threshold(canny_edges, low, high, weak)
    canny_edges = hysteresis(canny_edges, weak)
    return canny_edges, gradient_magnitude
