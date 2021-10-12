import argparse as ap
import numpy as np

import imageio as imo
import matplotlib.pyplot as plt
from Inpainter import Inpainter


def main():
    """ Usage in inpainting directory :

        python Code/main.py -i Data/Image.png -m Data/Mask.png -o Data/Output.png

         To make a GIF from result, go to https://ezgif.com/maker and use images in '/Data/Temp' folder """

    args = parse_arguments()

    image = imo.imread(args.input)
    mask = imo.imread(args.mask)
    output_filepath = args.output

    # convert image in rgb and mask in gray scale
    # image = rgba2rgb(image)
    mask = rgba2gray(mask)

    output_image = Inpainter(image, mask).inpaint()
    imo.imwrite(output_filepath, output_image)
    show_image(output_image, 'result')


def parse_arguments():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        help='the filepath to the image containing object to be edited',
        default='../Data/Baseball.jpg'
    )

    parser.add_argument(
        '-m',
        '--mask',
        help='the mask of the region to be removed',
        default='../Data/Baseball_mask.jpg'
    )

    parser.add_argument(
        '-o',
        '--output',
        help='the filepath to save the output image',
        default='../Data/Baseball_output_2.jpg'
    )

    return parser.parse_args()


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def rgba2gray(rgba):
    gray = np.dot(rgba[..., :3], [0.299, 0.587, 0.114])
    return np.asarray(gray, dtype='uint8')


def show_image(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()
