import cv2
import argparse
import os
import numpy as np
from reduce_colors import reduce_colors_knn
from canny_edges import detect_canny_edges
from moviepy.editor import ImageClip, VideoFileClip


def cartoonify(reduced_colors, canny_edges, gradient_mask, border_color=[0,0,0]):
    rows = reduced_colors.shape[0]
    cols = reduced_colors.shape[1]
    cartoon = reduced_colors.copy()

    
    # apply sobel gradient mask
    cartoon = ImageClip(cartoon).set_mask(ImageClip(gradient_mask, ismask=True)).img

    # apply canny edges
    for r in range(rows):
        for c in range(0, cols):
            if canny_edges[r][c] > 0:
                cartoon[r][c] = border_color

    return cartoon


def cartoonify_image(img):
    reduced_colors = reduce_colors_knn(img, K=9, threshold=0.1)
    canny_edges, gradient_mask = detect_canny_edges(img)
    return cartoonify(reduced_colors, canny_edges, gradient_mask)


def cartoonify_video(frame):
    # use OpenCV function of kmeans for performance reasons
    # convert to np.float32
    Z = frame.reshape((-1,3))
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 9
    _,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    reduced_colors = res.reshape((frame.shape))
    canny_edges, gradient_mask = detect_canny_edges(frame)
    return cartoonify(reduced_colors, canny_edges, gradient_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input file path")
    parser.add_argument("-o", "--output", type=str, help="output file path - defaults to ../cartoon_input.png", default="")
    parser.add_argument("-v", "--video", action="store_true", help="input file is a video")
    args = parser.parse_args()

    input_path = str(args.input)
    input_file_name = os.path.basename(input_path)
    output_path = args.output if args.output else f"{input_path.removesuffix(input_file_name)}cartoon_{input_file_name}"

    print('==================================================================================')
    print('PSU CS 410/510 - Computer Vision, Winter 2022, Final Project: Cartoonifying Images')
    print('==================================================================================\n')

    if args.video:
        print('This process will take a long while...')
        with VideoFileClip(input_path) as video:
            # ===== perform operation
            with video.fl_image(cartoonify_video) as cartoon:
                cartoon.write_videofile(output_path)
    else:
        print('This process can take a few moments...')
        # assume image
        # ===== read input image
        img = cv2.imread(input_path, flags=cv2.IMREAD_COLOR)

        # ===== perform operation
        cartoon = cartoonify_image(img)

        # ===== save generated cartoon
        cv2.imwrite(filename=output_path, img=(cartoon.clip(0.0, 255.0)))

    print(f'Successfully Cartoonified {input_path} to {output_path}')
