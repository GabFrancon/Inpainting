import argparse as ap
import numpy as np
import os
import imageio as imo
import matplotlib.pyplot as plt
from Inpainter import Inpainter
from Corrector import Corrector


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

    output_image, shift_map = Inpainter(image, mask).inpaint()
    imo.imwrite(output_filepath, output_image)

    final_image = Corrector(output_image, mask, shift_map).correct()
    imo.imwrite('../Data/Panel_output_final.jpg', final_image)
    show_image(final_image, 'final image')


def parse_arguments():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        help='the filepath to the image containing object to be edited',
        default='../Data/Panel_output_CxD.jpg'
    )

    parser.add_argument(
        '-m',
        '--mask',
        help='the mask of the region to be removed',
        default='../Data/Mask/Panel_mask_2.jpg'
    )

    parser.add_argument(
        '-o',
        '--output',
        help='the filepath to save the output image',
        default='../Data/Panel_output_CxD_2.jpg'
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
    directory1 = '../Data/Temp'
    for file in os.listdir(directory1):
        os.remove(os.path.join(directory1, file))
    directory2 = '../Data/Temp_corr'
    for file in os.listdir(directory2):
        os.remove(os.path.join(directory2, file))


def show_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()
