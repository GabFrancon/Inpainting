import time
import numpy as np
import matplotlib.pyplot as plt
import imageio as imo
from scipy import ndimage


class Inpainter:
    def __init__(self, image, mask):
        self.mask = np.copy(mask)
        self.fix_mask = np.copy(self.mask)
        self.image = np.copy(image)
        self.source_region = None
        self.fill_front = None
        self.priority = None
        self.confidence = None
        self.data = None
        self.iteration = 1
        self.patch_size = 9
        self.start_time = time.perf_counter()

        # Safety for the program to finish :
        # We make sure we don't make more loops than the total of number of pixels in the image
        self.safetyCount = 0
        self.safetyMax = None

    def inpaint(self):
        self.validate_inputs()
        self.initialize_attributes()
        done = False

        while not done:
            print(str(self.iteration) + ' :')
            self.identify_fill_front()
            self.compute_priorities()

            target_pixel, target_location, target_data = self.get_target()
            local_mask = self.get_local_mask(target_location)
            source_data = self.get_source(target_location, target_data, local_mask)

            self.update_image(target_location, target_data, source_data, local_mask)
            self.change_local_conf(target_pixel, target_location, local_mask)

            done = self.is_finished()

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
        self.save_temp_image()

    def identify_fill_front(self):
        laplacian = ndimage.laplace(self.mask)
        self.fill_front = (laplacian > 0).astype('uint8')
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
                print('\nmax pixel is None, new patch size = '+str(self.patch_size)+'\n')

        return max_pixel

    def is_patchable(self, pixel):
        half_size = (self.patch_size - 1) // 2
        height, width = self.image.shape[:2]
        return (half_size < pixel[0] < height - half_size) and (half_size < pixel[1] < width - half_size)

    def get_target(self):
        pixel = self.find_highest_priority()
        indices = self.patch_indices(pixel)
        data = self.patch_data(indices, self.image)

        print('target pixel : ' + str(pixel))
        # self.show_patch(self.image, indices, 'target patch')
        return pixel, indices, data

    def get_local_mask(self, indices):
        return self.patch_data(indices, self.mask)

    def get_source(self, target_location, target_data, local_mask):
        best_pixel = None
        best_match = None
        best_location = None
        distance = np.inf

        for pixel in self.source_region:

            source_patch = self.patch_indices(pixel)
            euclidian_dist = np.sqrt((target_location[0][0] - source_patch[0][0]) ** 2
                                     + (target_location[1][0] - source_patch[1][0]) ** 2)
            if euclidian_dist < 100:
                source_data = self.patch_data(source_patch, self.image)
                new_distance = self.calculate_distance(target_data, source_data, local_mask)

                if new_distance < distance:
                    distance = new_distance
                    best_pixel = pixel
                    best_location = source_patch
                    best_match = source_data

        print('source pixel : ' + str(best_pixel))
        print('distance : ' + str(distance))
        # self.show_patch(self.image, best_location, 'source patch')
        return best_match

    def update_image(self, patch, target_data, source_data, local_mask):
        [minX, maxX], [minY, maxY] = patch
        new_data = self.create_new_data(target_data, source_data, local_mask)

        self.image[minX:maxX, minY:maxY] = new_data
        self.mask[minX:maxX, minY:maxY] = 0
        # self.show_image(self.image, 'updated image')

        self.iteration += 1

    def change_local_conf(self, pixel, patch, mask):
        [minX, maxX], [minY, maxY] = patch
        old_conf = self.patch_data(patch, self.confidence)
        new_conf = np.zeros_like(old_conf)

        new_conf[mask == 0] = old_conf[mask == 0]
        new_conf[mask == 1] = self.confidence[pixel[0], pixel[1]]

        self.confidence[minX:maxX, minY:maxY] = new_conf

    def is_finished(self):
        white_number = self.mask.sum()
        self.safetyCount += 1

        if white_number == 0 or (self.safetyCount > self.safetyMax):
            print('total process time : ' + self.get_chrono())
            self.save_temp_image()

            return True
        else:
            prediction = round(white_number/(self.patch_size * self.patch_size)*3.92)
            print('remains approximately ' + str(prediction) + ' iterations')
            print('process time : ' + self.get_chrono())

            if self.iteration % 5 == 0:
                self.save_temp_image()

            return False

    def delimit_source_region(self):
        print('\ndelimiting source region...')
        height, width = self.mask.shape
        size = (self.patch_size - 1) // 2
        structured_elem_1 = np.ones((self.patch_size, self.patch_size))
        structured_elem_2 = np.ones((self.patch_size * 4, self.patch_size * 4))
        dilated_mask_1 = np.zeros_like(self.mask)
        dilated_mask_2 = np.zeros_like(self.mask)

        for i in range(height):
            for j in range(width):
                dilated_mask_1[i, j] = self.pixel_dilatation(self.mask, i, j, structured_elem_1)
                dilated_mask_2[i, j] = self.pixel_dilatation(self.mask, i, j, structured_elem_2)

        source_mask = dilated_mask_2 - dilated_mask_1
        source_mask[:size, :] = 0
        source_mask[:, :size] = 0
        source_mask[height - size:height, :] = 0
        source_mask[:, width - size:width] = 0

        source_region = np.copy(self.image)
        source_region[source_mask == 0] = 0
        source_region[self.mask == 1] = 255
        print("time to delimit the source region : " + self.get_chrono())
        self.show_image(source_region, 'source region')
        return np.argwhere(source_mask == 1)

    def get_chrono(self):
        return str(round(time.perf_counter() - self.start_time)) + ' seconds\n'

    def save_temp_image(self):
        # save the result of the last iteration
        imo.imwrite('../Data/Temp/' + str(self.iteration) + '.jpg', self.image)

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
    def pixel_dilatation(image, line, column, structured_elem):
        width = structured_elem.shape[0] // 2
        height = structured_elem.shape[1] // 2
        pixel_value = False

        for i in range(0, structured_elem.shape[0]):
            for j in range(0, structured_elem.shape[1]):
                x_image = line + i - width
                y_image = column + j - height
                if (x_image >= 0) and (x_image < image.shape[0]) and (y_image >= 0) and (y_image < image.shape[1]):
                    if image[x_image, y_image] and structured_elem[i, j]:
                        pixel_value = True

        return pixel_value

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
