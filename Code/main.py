import argparse as ap
import imageio as imo
from Inpainter import Inpainter


def main():
    args = parse_arguments()
    image = imo.imread(args.input)
    mask = imo.imread(args.mask)
    output_filepath = args.output

    output_image = Inpainter(image, mask).inpaint()
    imo.imwrite(output_filepath, output_image[:, :, 0])


def parse_arguments():
    parser = ap.ArgumentParser()
    parser.add_argument\
    (
        '-i',
        '--input',
        help='the filepath to the image containing object to be edited'
    )

    parser.add_argument\
    (
        '-m',
        '--mask',
        help='the mask of the region to be removed'
    )

    parser.add_argument\
    (
        '-o',
        '--output',
        help='the filepath to save the output image'
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
