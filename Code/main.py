import argparse as ap
import numpy as np
import os

import imageio as imo
import matplotlib.pyplot as plt
from Inpainter import Inpainter


def main():
    """ Usage in inpainting directory :

        python Code/main.py -i Data/Image.jpg -m Data/Mask.jpg -o Data/Output.jpg

         To make a GIF from result, go to https://ezgif.com/maker and use images in '/Data/Temp' folder """

    clear_temp_directory()
    args = parse_arguments()

    image = imo.imread(args.input)
    mask = imo.imread(args.mask)
    output_filepath = args.output

    # prepare image and mask
    mask = format_mask(mask)
    image = format_image(image, mask)

    # show_image(image, 'starter image')
    # show_image(mask, 'starter mask')

    output_image = Inpainter(image, mask).inpaint()
    imo.imwrite(output_filepath, output_image)
    # create_gif()
    show_image(output_image, 'result')


def parse_arguments():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        help='the filepath to the image containing object to be edited',
        default='../Data/Island.jpg'
    )

    parser.add_argument(
        '-m',
        '--mask',
        help='the mask of the region to be removed',
        default='../Data/Mask/Island_mask.jpg'
    )

    parser.add_argument(
        '-o',
        '--output',
        help='the filepath to save the output image',
        default='../Data/Island_output_CxD_2.jpg'
    )

    return parser.parse_args()


def format_mask(mask):

    mask = np.dot(mask[..., :3], [0.299, 0.587, 0.114])
    mask = np.asarray(mask, dtype='uint8')
    mask = (mask > 128).astype('float')
    return mask


def format_image(image, mask):
    if image.ndim < 3:
        image = image.reshape(image.shape[0], image.shape[1], 1).repeat(3, axis=2)
    image[mask == 1] = 255
    return image


def clear_temp_directory():
    directory = '../Data/Temp'
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))


def create_gif():
    directory = '../Data/Temp'
    with imo.get_writer('../Data/Process/process.gif', mode='I') as writer:
        for file in os.listdir(directory):
            image = imo.imread(os.path.join(directory, file))
            writer.append_data(image)


def show_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()
