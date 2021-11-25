import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation


def patch_data(indices, source):
    # returns the pixel values of the given indices in source
    [minX, maxX], [minY, maxY] = indices
    return source[minX:maxX, minY:maxY]


def make_dilation(mask, thickness, offset=0):
    # returns a peel of given thickness around the mask
    structured_elem = np.ones((thickness + offset, thickness + offset))
    second_mask = mask

    if offset != 0:
        offset_elem = np.ones((offset, offset))
        second_mask = binary_dilation(mask, offset_elem).astype('uint8')

    dilated_mask = binary_dilation(mask, structured_elem).astype('uint8')
    return (dilated_mask - second_mask).astype('uint8')


def get_chrono(start_time):
    # returns the time since the process started
    return str(round(time.perf_counter() - start_time)) + ' seconds'


def calculate_distance(target_color, source_color, mask):
    # returns the squared color and texture dist between pixels of given target and source
    color_dist = np.zeros_like(target_color)
    color_dist[mask == 0] = np.square(source_color[mask == 0] - target_color[mask == 0])
    return color_dist.sum()


def create_new_data(target_data, source_data, local_mask):
    # returns a patch composed of already known target pixels and new pixels from source patch
    new_data = np.zeros_like(target_data)
    new_data[local_mask == 0] = target_data[local_mask == 0]
    new_data[local_mask == 1] = source_data[local_mask == 1]
    return new_data


def show_image(image, title, axis):
    # displays an image with a title
    axis.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def show_patch(source, indices, color):
    # shows the patch on the given source
    [minX, maxX], [minY, maxY] = indices
    image = np.copy(source)

    image[minX - 1:maxX, minY - 1] = color
    image[minX - 1:maxX, maxY] = color
    image[minX - 1, minY - 1:maxY] = color
    image[maxX, minY - 1:maxY + 1] = color
    return image


def show_vectors(mask, vertices, color, axis):
    origins = np.argwhere(mask > 0)
    sorted_vertices = np.array([vertices[p[0], p[1]] for p in origins])

    axis.quiver(origins[:, 1],
                origins[:, 0],
                sorted_vertices[:, 1],
                sorted_vertices[:, 0],
                color=color,
                scale=1,
                scale_units='xy',
                angles='xy')


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=60):
    # Displays a terminal progress bar
    """
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
