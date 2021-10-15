import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio as imo
from scipy import ndimage
from skimage.color import rgb2hsv
from skimage.morphology import binary_dilation


class Inpainter:
    def __init__(self, image, mask):
        self.mask = np.copy(mask)
        self.image = np.copy(image)
        self.hsv_image = rgb2hsv(self.image)
        self.source_region = None
        self.fill_front = None
        self.priority = None
        self.confidence = None
        self.data = None
        self.iteration = 1
        self.patch_size = 9
        self.pixel_to_fill = self.mask.sum()
        self.start_time = time.perf_counter()

        # Safety for the program to finish :
        # We make sure we don't make more loops than the total of number of pixels in the image
        self.safetyCount = 0
        self.safetyMax = None

    def inpaint(self):
        self.validate_inputs()
        self.initialize_attributes()
        done = False

        print('\ninpainting image...')
        self.save_temp_image(True)

        while not done:
            self.identify_fill_front()
            self.compute_priorities()

            target_pixel, target_location, target_data = self.get_target()
            local_mask = self.get_local_mask(target_location)
            source_data = self.get_source(target_pixel, target_location, local_mask)

            self.update_image(target_location, target_data, source_data, local_mask)
            self.change_local_conf(target_pixel, target_location, local_mask)

            done = self.is_finished()

        print('\n\ntotal process time : ' + self.get_chrono())
        self.save_temp_image(True)
        return self.image

    def validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def initialize_attributes(self):
        height, width = self.image.shape[:2]

        self.source_region = self.delimit_source_region()
        self.confidence = (1 - self.mask)
        self.data = np.zeros([height, width])
        self.safetyMax = height * width

    def delimit_source_region(self):
        height, width = self.mask.shape
        size = (self.patch_size - 1) // 2
        source_mask = self.make_dilation(self.patch_size * 5)

        source_mask[:size, :] = 0
        source_mask[:, :size] = 0
        source_mask[height - size:height, :] = 0
        source_mask[:, width - size:width] = 0

        args = np.argwhere(source_mask == 1)
        for pixel in args:
            indices = self.patch_indices(pixel)
            data = self.patch_data(indices, self.mask)
            if data.sum() != 0:
                source_mask[pixel[0], pixel[1]] = 0

        return np.argwhere(source_mask == 1)

    def identify_fill_front(self):
        # laplacian = ndimage.laplace(self.mask)
        # self.fill_front = (laplacian > 0).astype('uint8')
        self.fill_front = self.make_dilation(3)
        # self.show_image(self.fill_front, 'fill front')

    def compute_priorities(self):
        self.update_confidence()
        self.update_data()
        self.priority = self.confidence * self.data

    def update_confidence(self):
        new_confidence = np.copy(self.confidence)
        edge = np.argwhere(self.fill_front > 0)

        for pixel in edge:
            indices = self.patch_indices(pixel)
            confidence_values = self.patch_data(indices, self.confidence)

            val = sum(sum(confidence_values)) / (self.patch_size * self.patch_size)
            new_confidence[pixel[0], pixel[1]] = val

        self.confidence = new_confidence

    def update_data(self):
        self.data = 1

    def find_highest_priority(self):
        edge = np.argwhere(self.fill_front > 0)
        max_priority = -np.inf
        max_pixel = None

        while max_pixel is None:
            for pixel in edge:
                if self.is_patchable(pixel):
                    priority = self.priority[pixel[0]][pixel[1]]
                    if priority > max_priority:
                        max_priority = priority
                        max_pixel = pixel
            if max_pixel is None:
                self.patch_size = self.patch_size // 2

        return max_pixel

    def get_target(self):
        pixel = self.find_highest_priority()
        indices = self.patch_indices(pixel)
        data = self.patch_data(indices, np.copy(self.image))

        # self.show_patch(self.image, indices, 'target patch')
        return pixel, indices, data

    def get_local_mask(self, indices):
        return self.patch_data(indices, np.copy(self.mask))

    def get_source(self, target_pixel, target_location, local_mask):
        best_location = None
        distance = np.inf
        max_dist = 100
        hsv_target = self.patch_data(target_location, self.hsv_image)

        while best_location is None:

            for pixel in self.source_region:

                euclidian_dist = np.linalg.norm(target_pixel - pixel)
                if euclidian_dist < max_dist:
                    source_location = self.patch_indices(pixel)
                    hsv_source = self.patch_data(source_location, self.hsv_image)
                    new_distance = self.calculate_distance(hsv_target, hsv_source, local_mask)

                    if new_distance < distance:
                        distance = new_distance
                        best_location = source_location

            if best_location is None:
                max_dist += 10

        # self.show_patch(self.image, best_location, 'source patch')
        return self.patch_data(best_location, np.copy(self.image))

    def update_image(self, patch, target_data, source_data, local_mask):
        [minX, maxX], [minY, maxY] = patch
        new_data = self.create_new_data(target_data, source_data, local_mask)

        self.image[minX:maxX, minY:maxY] = new_data
        self.hsv_image[minX:maxX, minY:maxY] = rgb2hsv(new_data)
        self.mask[minX:maxX, minY:maxY] = 0
        # self.show_image(self.image, 'updated image')
        self.iteration += 1

    def change_local_conf(self, pixel, patch, mask):
        [minX, maxX], [minY, maxY] = patch
        old_conf = self.patch_data(patch, np.copy(self.confidence))
        new_conf = np.zeros_like(old_conf)

        new_conf[mask == 0] = old_conf[mask == 0]
        new_conf[mask == 1] = self.confidence[pixel[0], pixel[1]]
        self.confidence[minX:maxX, minY:maxY] = new_conf

    def is_finished(self):
        white_number = self.mask.sum()
        self.safetyCount += 1
        self.print_progress(self.pixel_to_fill - white_number, self.pixel_to_fill)
        self.save_temp_image()
        return white_number == 0 or (self.safetyCount > self.safetyMax)

    def is_patchable(self, pixel):
        half_size = (self.patch_size - 1) // 2
        height, width = self.image.shape[:2]
        return (half_size < pixel[0] < height - half_size) and (half_size < pixel[1] < width - half_size)

    def make_dilation(self, thickness):
        structured_elem = np.ones((thickness, thickness))
        dilated_mask = binary_dilation(self.mask, structured_elem).astype('uint8')
        return (dilated_mask - self.mask).astype('uint8')

    def get_chrono(self):
        return str(round(time.perf_counter() - self.start_time)) + ' seconds\n'

    def save_temp_image(self, force=False):
        # save the result of the last iteration
        if self.iteration % 5 == 0 or force:
            copy = np.copy(self.image)
            self.identify_fill_front()
            copy[self.fill_front > 0] = [157, 47, 186]
            imo.imwrite('../Data/Temp/' + str(self.iteration) + '.png', copy)

    def patch_indices(self, pixel):
        # returns the indices (min and max) of the patch centered around the given pixel
        half_size = (self.patch_size - 1) // 2
        height, width = self.image.shape[:2]

        min_x = max(0, pixel[0] - half_size)
        max_x = min(pixel[0] + half_size, height - 1)
        min_y = max(0, pixel[1] - half_size)
        max_y = min(pixel[1] + half_size, width - 1)

        return [min_x, max_x + 1], [min_y, max_y + 1]

    @staticmethod
    def patch_data(indices, source):
        # returns the pixel values of the given indices in source
        [minX, maxX], [minY, maxY] = indices
        return source[minX:maxX, minY:maxY]

    @staticmethod
    def calculate_distance(target_data, source_data, mask):
        # returns the squared distance between pixels of given target and source
        diff = np.zeros_like(target_data)
        diff[mask == 0] = (source_data[mask == 0] - target_data[mask == 0]) ** 2
        return diff.sum()

    @staticmethod
    def create_new_data(target_data, source_data, local_mask):
        # returns a patch composed of already known target pixels and new pixels from source patch
        new_data = np.zeros_like(target_data)
        new_data[local_mask == 0] = target_data[local_mask == 0]
        new_data[local_mask == 1] = source_data[local_mask == 1]
        return new_data

    @staticmethod
    def show_image(image, title):
        # method to display an image with a title
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()

    @staticmethod
    def show_patch(source, indices, title):
        # method to display the location of the patch on the given source
        [minX, maxX], [minY, maxY] = indices
        image = np.copy(source)

        image[minX:maxX, minY] = [255, 0, 0]
        image[minX:maxX, maxY] = [255, 0, 0]
        image[minX, minY:maxY] = [255, 0, 0]
        image[maxX, minY:maxY] = [255, 0, 0]

        plt.imshow(image)
        plt.title(title)
        plt.show()

    @staticmethod
    def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=60):
        """
        Call in a loop to create terminal progress bar
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
