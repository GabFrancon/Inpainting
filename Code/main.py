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

    # to convert the mask in gray scale
    mask = np.dot(mask[..., :3], [0.2989, 0.5870, 0.1140])

    output_image = Inpainter(image, mask).inpaint()
    imo.imwrite(output_filepath, output_image)
    show_image(output_image)


def parse_arguments():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        help='the filepath to the image containing object to be edited',
        default='../Data/Image.png'
    )

    parser.add_argument(
        '-m',
        '--mask',
        help='the mask of the region to be removed',
        default='../Data/Mask2.png'
    )

    parser.add_argument(
        '-o',
        '--output',
        help='the filepath to save the output image',
        default='../Data/Output.png'
    )

    return parser.parse_args()


def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
