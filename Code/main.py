import argparse as ap
import imageio as imo
import matplotlib.pyplot as plt
from Inpainter import Inpainter


def main():
    """ Usage in inpainting directory :

        python Code/main.py -i Data/Image.png -m Data/Mask.png -o Data/Output.png """

    args = parse_arguments()

    image = imo.imread(args.input)
    mask = imo.imread(args.mask)
    output_filepath = args.output

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
        default='../Data/Mask.png'
    )

    parser.add_argument(
        '-o',
        '--output',
        help='the filepath to save the output image',
        default='../Data/Output.png'
    )

    return parser.parse_args()


def show_image(image):
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()
