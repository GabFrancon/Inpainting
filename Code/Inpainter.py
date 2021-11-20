import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio as imo
from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import binary_dilation
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter


class Inpainter:
    def __init__(self, image, mask):
        # inputs
        self.mask = np.copy(mask)
        self.image = np.copy(image)

        # data associated with pixels
        self.source_region = None
        self.shift_map     = None
        self.fill_front    = None
        self.priority      = None
        self.confidence    = None
        self.data          = None
        self.normal        = None
        self.isophote      = None
        self.hsv_image     = None
        self.gray_image    = None

        # parameters relative to target and source region
        self.patch_size         = 7
        self.source_region_size = 4
        self.last_target        = None

        # constants and variables used to display info during process
        self.iteration     = None
        self.pixel_to_fill = None
        self.start_time    = None

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

        print('\ntotal process time : ' + self.get_chrono())
        print('total iterations   : ' + str(self.iteration))
        self.save_temp_image(True)
        return self.image, self.shift_map

    def validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def initialize_attributes(self):
        h, w = self.mask.shape

        self.source_region = self.delimit_source_region()
        self.shift_map     = self.initialize_shift_map()
        self.confidence    = 1 - self.mask
        self.data          = np.zeros_like(self.mask).astype('float')
        self.normal        = np.zeros((h, w, 2))
        self.isophote      = np.zeros((h, w, 2))
        self.hsv_image     = rgb2hsv(self.image)
        self.gray_image    = rgb2gray(self.image)
        self.iteration     = 1
        self.pixel_to_fill = self.mask.sum()
        self.last_target   = [0, self.image.shape[0]], [0, self.image.shape[1]]
        self.start_time    = time.perf_counter()

    def delimit_source_region(self):
        h, w = self.mask.shape
        size = (self.patch_size - 1) // 2
        source_mask = self.make_dilation(self.patch_size * self.source_region_size, self.patch_size)

        source_mask[:size, :] = 0
        source_mask[:, :size] = 0
        source_mask[h - size:h, :] = 0
        source_mask[:, w - size:w] = 0
        return np.argwhere(source_mask == 1)

    def initialize_shift_map(self):
        h, w = self.mask.shape
        shift_map = np.zeros((h, w, 2))
        for i in range(h):
            for j in range(w):
                if self.mask[i, j] == 0:
                    shift_map[i, j] = [i, j]
        return shift_map

    def identify_fill_front(self):
        self.fill_front = self.make_dilation(3)

    def compute_priorities(self):
        self.update_confidence()
        self.update_data()
        self.priority = self.data*self.confidence

    def average_patch(self, pixel, source):
        patch = self.patch_indices(pixel)
        data = self.patch_data(patch, source)
        return sum(sum(data)) / (self.patch_size * self.patch_size)

    def update_confidence(self):
        new_confidence = np.copy(self.confidence)
        edge = np.argwhere(self.fill_front > 0)

        for p in edge:
            val = self.average_patch(p, self.confidence)
            new_confidence[p[0], p[1]] = val

        self.confidence = new_confidence

    def update_data(self):
        normal = self.compute_normal()
        isophote, iso_norm = self.compute_isophote()

        self.normal[self.fill_front == 0] = 0
        self.isophote[self.fill_front == 0] = 0
        self.data[self.fill_front == 0] = 0

        edge = np.argwhere(self.fill_front > 0)
        [min_X, max_X], [min_Y, max_Y] = self.last_target
        half = (self.patch_size - 1) // 2

        for p in edge:
            if (min_X-half <= p[0] < max_X+half) and (min_Y-half <= p[1] < max_Y+half):
                patch = self.patch_indices(p)
                data = self.patch_data(patch, iso_norm)
                mask = binary_dilation(self.get_local_mask(patch), np.ones((6, 6))).astype('uint8')
                data[mask == 1] = 0
                x, y = np.unravel_index(data.argmax(), data.shape)

                p_isophote = isophote[patch[0][0]+x, patch[1][0]+y]
                p_normal = normal[p[0], p[1]]

                self.normal[p[0], p[1]] = p_normal
                self.isophote[p[0], p[1]] = p_isophote
                self.data[p[0], p[1]] = abs(np.dot(p_normal, p_isophote))

        """axis = self.config_plot()
        self.show_vectors(self.fill_front, 5*self.normal, 'r', axis)
        self.show_vectors(self.fill_front, 50*self.isophote, 'b', axis)
        self.show_image(self.image, '[red]normal and [blue]isophote vertices', axis)"""

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
        return normal

    def compute_isophote(self):
        gauss_img = gaussian_filter(self.gray_image, 1)
        gradX, gradY = np.array(np.gradient(gauss_img))

        isophote = np.dstack((-gradY, gradX))
        norm = np.sqrt(np.square(gradX) + np.square(gradY))
        return isophote, norm

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

        return pixel, indices, data

    def get_local_mask(self, indices):
        return self.patch_data(indices, np.copy(self.mask))

    def get_source(self, target_pixel, target_location, local_mask):
        best_pixel = None
        best_location = None
        best_distance = np.inf
        hsv_target = self.patch_data(target_location, self.hsv_image)

        while best_location is None:

            for pixel in self.source_region:
                source_location = self.patch_indices(pixel)
                hsv_source = self.patch_data(source_location, self.hsv_image)
                distance = self.calculate_distance(hsv_target, hsv_source, local_mask)
                if distance < best_distance:
                    best_distance = distance
                    best_pixel = pixel
                    best_location = source_location

        '''patch = self.show_patch(self.image, target_location, [255, 0, 0])
        patch = self.show_patch(patch, best_location, [0, 0, 255])
        self.show_image(patch, '[red]target and [blue]source patches', self.config_plot())'''

        [minX, maxX], [minY, maxY] = target_location
        new_shift = self.patch_data(target_location, self.shift_map)
        new_shift[local_mask == 1] = best_pixel - target_pixel
        self.shift_map[minX:maxX, minY:maxY] = new_shift
        return self.patch_data(best_location, np.copy(self.image))

    def update_image(self, patch, target_data, source_data, local_mask):
        [minX, maxX], [minY, maxY] = patch
        new_data = self.create_new_data(target_data, source_data, local_mask)

        self.image[minX:maxX, minY:maxY] = new_data
        self.hsv_image[minX:maxX, minY:maxY] = rgb2hsv(new_data)
        self.gray_image[minX:maxX, minY:maxY] = rgb2gray(new_data)
        self.mask[minX:maxX, minY:maxY] = 0

    def change_local_conf(self, pixel, patch, mask):
        [minX, maxX], [minY, maxY] = patch
        old_conf = self.patch_data(patch, np.copy(self.confidence))
        new_conf = np.zeros_like(old_conf)

        new_conf[mask == 0] = old_conf[mask == 0]
        new_conf[mask == 1] = self.confidence[pixel[0], pixel[1]]
        self.confidence[minX:maxX, minY:maxY] = new_conf

    def is_finished(self):
        pixels_remaining = self.mask.sum()
        self.iteration += 1
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

    def config_plot(self):
        fig, ax = plt.subplots(1)
        ax.set_xlim((0, self.image.shape[1]))
        ax.set_ylim((0, self.image.shape[0]))
        ax.invert_yaxis()
        plt.xlabel('y')
        plt.ylabel('x')
        return ax

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
    def calculate_distance(target_color, source_color, mask):
        # returns the squared color and texture dist between pixels of given target and source
        """targetH = target_color[:, :, 0]
        targetS = target_color[:, :, 1]
        targetV = target_color[:, :, 2]
        sourceH = source_color[:, :, 0]
        sourceS = source_color[:, :, 1]
        sourceV = source_color[:, :, 2]

        diffS = np.zeros_like(targetS)
        diffV = np.zeros_like(targetV)
        minH = np.zeros_like(targetH)
        minH[mask == 0] = np.abs(sourceH[mask == 0] - targetH[mask == 0])
        minH[minH > 0.5] = 1 - minH[minH > 0.5]
        diffH = minH ** 2
        diffS[mask == 0] = (sourceS[mask == 0] - targetS[mask == 0]) ** 2
        diffV[mask == 0] = (sourceV[mask == 0] - targetV[mask == 0]) ** 2

        color_dist = np.dstack((diffH, diffS, diffV))"""
        color_dist = np.zeros_like(target_color)
        color_dist[mask == 0] = (source_color[mask == 0] - target_color[mask == 0]) ** 2
        return color_dist.sum()

    @staticmethod
    def create_new_data(target_data, source_data, local_mask):
        # returns a patch composed of already known target pixels and new pixels from source patch
        new_data = np.zeros_like(target_data)
        new_data[local_mask == 0] = target_data[local_mask == 0]
        new_data[local_mask == 1] = source_data[local_mask == 1]
        return new_data

    @staticmethod
    def show_image(image, title, axis):
        # displays an image with a title
        axis.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()

    @staticmethod
    def show_patch(source, indices, color):
        # shows the patch on the given source
        [minX, maxX], [minY, maxY] = indices
        image = np.copy(source)

        image[minX-1:maxX, minY-1] = color
        image[minX-1:maxX, maxY] = color
        image[minX-1, minY-1:maxY] = color
        image[maxX, minY-1:maxY+1] = color
        return image

    @staticmethod
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
