import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio as imo
from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import binary_dilation
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import convolve


class Inpainter:
    def __init__(self, image, mask):
        # inputs
        self.mask = np.copy(mask)
        self.image = np.copy(image)

        # data associated with pixels
        self.source_region = None
        self.fill_front = None
        self.priority = None
        self.confidence = None
        self.data = None
        self.hsv_image = None

        # parameters relative to target and source region
        self.patch_size = 9
        self.source_region_size = 6
        self.max_distance = np.inf
        self.last_target = None

        # constants and variables used to display info during process
        self.iteration = None
        self.pixel_to_fill = None
        self.start_time = None

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
        self.source_region = self.delimit_source_region()
        self.confidence = 1 - self.mask
        self.data = np.zeros_like(self.mask).astype('float')
        self.hsv_image = rgb2hsv(self.image)

        self.iteration = 1
        self.pixel_to_fill = self.mask.sum()
        self.last_target = [0, self.image.shape[0]], [0, self.image.shape[1]]
        self.start_time = time.perf_counter()

    def delimit_source_region(self):
        height, width = self.mask.shape
        size = (self.patch_size - 1) // 2
        source_mask = self.make_dilation(self.patch_size * self.source_region_size, self.patch_size)

        source_mask[:size, :] = 0
        source_mask[:, :size] = 0
        source_mask[height - size:height, :] = 0
        source_mask[:, width - size:width] = 0

        return np.argwhere(source_mask == 1)

    def identify_fill_front(self):
        self.fill_front = self.make_dilation(3)
        # self.show_image(self.fill_front, 'fill front')

    def compute_priorities(self):
        self.update_confidence()
        self.update_data()
        self.priority = self.confidence * self.data
        # self.show_image(self.priority, 'C(p)xD(p)')

    def update_confidence(self):
        new_confidence = np.copy(self.confidence)
        edge = np.argwhere(self.fill_front > 0)

        for p in edge:
            indices = self.patch_indices(p)
            confidence_values = self.patch_data(indices, self.confidence)

            val = sum(sum(confidence_values)) / (self.patch_size * self.patch_size)
            new_confidence[p[0], p[1]] = val

        self.confidence = new_confidence
        # self.show_image(self.confidence, 'C(p)')

    def update_data(self):
        normal = self.compute_normal()
        isophote = self.compute_isophote()

        edge = np.argwhere(self.fill_front > 0)
        data_mask = self.make_dilation(3, 5)
        [minX, maxX], [minY, maxY] = self.last_target
        # vertices = []
        # origins = []

        for p in edge:
            if (minX <= p[0] < maxX) and (minY <= p[1] < maxY):
                one = np.ones_like(self.mask)
                one[p[0], p[1]] = 0
                distance_matrix = distance_transform_edt(one)
                # distance_matrix[np.linalg.norm(isophote, axis=2) > 1] = 1000
                # distance_matrix[self.mask == 1] = 1000
                distance_matrix[data_mask == 0] = 1000
                close_pixel = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)

                p_normal = normal[p[0], p[1]]
                p_isophote = isophote[close_pixel[0], close_pixel[1]]

                self.data[p[0], p[1]] = abs(np.dot(p_normal, p_isophote))

                '''vertices.append(5*p_normal)
                vertices.append(100*abs(p_isophote))
                origins.append(p)
                origins.append(close_pixel)

        self.show_vectors(self.image, np.array(origins), np.array(vertices),
                          ['r', 'b'], 'normal (red) and isophote (blue)', False)'''

        self.data /= np.max(self.data)
        # self.show_image(self.data, 'D(p)')

    def compute_normal(self):
        x_elem = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_elem = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        normalX = convolve(self.mask.astype(float), x_elem)
        normalY = convolve(self.mask.astype(float), y_elem)
        normal_value = np.dstack((-normalY, normalX))

        h, w = normal_value.shape[:2]
        norm = np.sqrt(normalX ** 2 + normalY ** 2).reshape(h, w, 1).repeat(2, axis=2)
        norm[norm == 0] = 1
        normal = normal_value / norm

        # self.show_vectors(self.image, self.fill_front, 5*normal, 'r', 'normal')
        return normal

    def compute_isophote(self):
        grey_image = rgb2gray(self.image)
        grey_image[self.fill_front > 0] = 1000
        gradient = np.array(np.gradient(grey_image))

        gradientX = gradient[0]
        gradientY = gradient[1]
        isophote = abs(np.dstack((-gradientY, gradientX)))

        # self.show_vectors(self.image, self.make_dilation(3, 3), 100*isophote, 'b', 'isophote')
        return isophote

    def avg_on_front(self, data):
        new_data = np.copy(data)
        edge = np.argwhere(self.fill_front > 0)
        for p in edge:
            indices = self.patch_indices(p)
            values = self.patch_data(indices, data)
            non_null = values[values > 0]
            avg = sum(sum(abs(values))) / non_null.size
            new_data[p[0], p[1]] = avg

        return new_data

    def find_highest_priority(self):
        edge = np.argwhere(self.fill_front > 0)
        max_priority = -np.inf
        max_pixel = None

        while max_pixel is None:
            for p in edge:
                if self.is_patchable(p):
                    priority = self.priority[p[0], p[1]]
                    if priority > max_priority:
                        max_priority = priority
                        max_pixel = p
            if max_pixel is None:
                self.patch_size = self.patch_size // 2

        return max_pixel

    def get_target(self):
        pixel = self.find_highest_priority()
        indices = self.patch_indices(pixel)
        data = self.patch_data(indices, np.copy(self.image))
        self.last_target = indices

        # self.show_patch(self.image, indices, 'target patch')
        return pixel, indices, data

    def get_local_mask(self, indices):
        return self.patch_data(indices, np.copy(self.mask))

    def get_source(self, target_pixel, target_location, local_mask):
        best_location = None
        distance = np.inf
        hsv_target = self.patch_data(target_location, self.hsv_image)

        while best_location is None:

            for pixel in self.source_region:

                euclidian_dist = np.linalg.norm(target_pixel - pixel)
                if euclidian_dist < self.max_distance:
                    source_location = self.patch_indices(pixel)
                    hsv_source = self.patch_data(source_location, self.hsv_image)
                    new_distance = self.calculate_distance(hsv_target, hsv_source, local_mask)

                    if new_distance < distance:
                        distance = new_distance
                        best_location = source_location

            if best_location is None:
                self.max_distance += 10

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
        pixels_remaining = self.mask.sum()
        self.print_progress(self.pixel_to_fill - pixels_remaining, self.pixel_to_fill)
        self.save_temp_image()
        return pixels_remaining == 0 or (self.iteration > self.pixel_to_fill)

    def is_patchable(self, pixel):
        # checks if the given pixel is not too close to the borders to be patchable
        half_size = (self.patch_size - 1) // 2
        height, width = self.image.shape[:2]
        return (half_size < pixel[0] < height - half_size) and (half_size < pixel[1] < width - half_size)

    def make_dilation(self, thickness, offset=0):
        # returns a peel of given thickness around the mask
        structured_elem = np.ones((thickness, thickness))
        mask = self.mask

        if offset != 0:
            offset_elem = np.ones((offset, offset))
            mask = binary_dilation(self.mask, offset_elem).astype('uint8')

        dilated_mask = binary_dilation(mask, structured_elem).astype('uint8')
        return (dilated_mask - mask).astype('uint8')

    def get_chrono(self):
        # returns the time since the process started
        return str(round(time.perf_counter() - self.start_time)) + ' seconds'

    def save_temp_image(self, force=False):
        # saves the result of the last iteration
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
        # displays an image with a title
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()

    @staticmethod
    def show_patch(source, indices, title):
        # shows the patch on the given source
        [minX, maxX], [minY, maxY] = indices
        image = np.copy(source)

        image[minX-1:maxX, minY-1] = [255, 0, 0]
        image[minX-1:maxX, maxY] = [255, 0, 0]
        image[minX-1, minY-1:maxY] = [255, 0, 0]
        image[maxX, minY-1:maxY+1] = [255, 0, 0]

        plt.imshow(image)
        plt.title(title)
        plt.show()

    @staticmethod
    def show_vectors(source, mask, coords, colors, title, with_mask=True):
        if with_mask:
            origins = np.argwhere(mask > 0)
            vectors = np.array([coords[p[0], p[1]] for p in origins])
        else:
            origins = mask
            vectors = coords

        fig, ax = plt.subplots(1)
        ax.set_xlim((0, source.shape[1]))
        ax.set_ylim((0, source.shape[0]))
        ax.invert_yaxis()
        plt.xlabel('y')
        plt.ylabel('x')

        ax.quiver(origins[:, 1],
                  origins[:, 0],
                  vectors[:, 1],
                  vectors[:, 0],
                  color=colors,
                  scale=1,
                  scale_units='xy',
                  angles='xy')

        plt.title(title)
        plt.imshow(source, cmap='gray')
        plt.show()

    @staticmethod
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
