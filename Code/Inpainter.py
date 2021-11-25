import imageio as imo
from skimage.color import rgb2hsv, rgb2gray
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from utils import *


class Inpainter:
    def __init__(self, image, mask):
        # inputs
        self.mask = np.copy(mask)
        self.image = np.copy(image)
        self.fix_img = image

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
        self.patch_size         = 9
        self.source_region_size = 3
        self.max_dist           = np.inf
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

        print('\ntotal process time : ' + get_chrono(self.start_time))
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
        source_mask = make_dilation(self.mask, self.patch_size * self.source_region_size, self.patch_size)

        source_mask[:size, :] = 0
        source_mask[:, :size] = 0
        source_mask[h - size:h, :] = 0
        source_mask[:, w - size:w] = 0
        copy = np.copy(self.image)
        copy[source_mask == 0] = 0

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
        self.fill_front = make_dilation(self.mask, 3)

    def compute_priorities(self):
        self.update_confidence()
        self.update_data()
        self.priority = self.data*self.confidence

    def average_patch(self, pixel, source):
        patch = self.patch_indices(pixel)
        data = patch_data(patch, source)
        return np.sum(data) / (self.patch_size * self.patch_size)

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

        elem = np.ones((12, 12))
        mask = binary_dilation(self.mask, elem).astype('uint8')
        iso_norm[mask == 1] = 0

        edge = np.argwhere(self.fill_front > 0)
        [min_X, max_X], [min_Y, max_Y] = self.last_target
        half = (self.patch_size - 1) // 2

        for p in edge:
            if (min_X-half <= p[0] < max_X+half) and (min_Y-half <= p[1] < max_Y+half):
                mat = np.ones_like(self.mask)
                mat[p[0], p[1]] = 0
                dist = distance_transform_edt(mat)
                dist[mask == 1] = np.inf
                pix = np.unravel_index(dist.argmin(), dist.shape)

                patch = self.patch_indices(pix)
                data = patch_data(patch, iso_norm)
                x, y = np.unravel_index(data.argmax(), data.shape)

                p_isophote = isophote[patch[0][0]+x, patch[1][0]+y]
                p_normal = normal[p[0], p[1]]

                self.normal[p[0], p[1]] = p_normal
                self.isophote[p[0], p[1]] = p_isophote
                self.data[p[0], p[1]] = abs(np.dot(p_normal, p_isophote))

        """show_vectors(self.fill_front, 5*self.normal, 'r', axis)
        axis = self.config_plot()
        show_vectors(self.fill_front, 50*self.isophote, 'r', axis)
        show_image(self.fix_img, '[red]normal and [blue]isophote vertices', axis)"""

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
        data = patch_data(indices, np.copy(self.image))
        self.last_target = indices

        return pixel, indices, data

    def get_local_mask(self, indices):
        return patch_data(indices, np.copy(self.mask))

    def get_source(self, target_pixel, target_location, local_mask):
        best_pixel = None
        best_location = None
        best_distance = np.inf
        hsv_target = patch_data(target_location, self.hsv_image)

        while best_location is None:

            for pixel in self.source_region:
                if np.linalg.norm(target_pixel - pixel) < self.max_dist:
                    source_location = self.patch_indices(pixel)
                    hsv_source = patch_data(source_location, self.hsv_image)
                    distance = calculate_distance(hsv_target, hsv_source, local_mask)

                    if distance < best_distance:
                        best_distance = distance
                        best_pixel = pixel
                        best_location = source_location
                if best_pixel is None:
                    self.max_dist += 5

        """patch = show_patch(self.image, target_location, [255, 0, 0])
        patch = show_patch(patch, best_location, [0, 0, 255])
        show_image(patch, '[red]target and [blue]source patches', self.config_plot())"""

        [minX, maxX], [minY, maxY] = target_location
        new_shift = patch_data(target_location, self.shift_map)
        new_shift[local_mask == 1] = best_pixel - target_pixel
        self.shift_map[minX:maxX, minY:maxY] = new_shift

        return patch_data(best_location, np.copy(self.image))

    def update_image(self, patch, target_data, source_data, local_mask):
        [minX, maxX], [minY, maxY] = patch
        new_data = create_new_data(target_data, source_data, local_mask)

        self.image[minX:maxX, minY:maxY] = new_data
        self.hsv_image[minX:maxX, minY:maxY] = rgb2hsv(new_data)
        self.gray_image[minX:maxX, minY:maxY] = rgb2gray(new_data)
        self.mask[minX:maxX, minY:maxY] = 0

    def change_local_conf(self, pixel, patch, mask):
        [minX, maxX], [minY, maxY] = patch
        old_conf = patch_data(patch, np.copy(self.confidence))
        new_conf = np.zeros_like(old_conf)

        new_conf[mask == 0] = old_conf[mask == 0]
        new_conf[mask == 1] = self.confidence[pixel[0], pixel[1]]
        self.confidence[minX:maxX, minY:maxY] = new_conf

    def is_finished(self):
        pixels_remaining = self.mask.sum()
        self.iteration += 1
        print_progress(self.pixel_to_fill - pixels_remaining, self.pixel_to_fill)
        self.save_temp_image()
        return pixels_remaining == 0 or (self.iteration > self.pixel_to_fill)

    def is_patchable(self, pixel):
        # checks if the given pixel is not too close to the borders to be patchable
        half_size = (self.patch_size - 1) // 2
        height, width = self.image.shape[:2]
        return (half_size < pixel[0] < height - half_size) and (half_size < pixel[1] < width - half_size)

    def save_temp_image(self, force=False):
        # saves the result of the last iteration
        if self.iteration % 15 == 0 or force:
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