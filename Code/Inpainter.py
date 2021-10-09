import time
import numpy as np
import matplotlib.pyplot as plt
import imageio as imo
from scipy import ndimage
import numexpr as ne


class Inpainter:
    def __init__(self, image, mask):
        self.mask = np.copy(mask.astype('uint8'))
        self.fix_mask = np.copy(mask.astype('uint8'))
        self.image = np.copy(image.astype('uint8'))
        self.fill_front = None
        self.priority = None
        self.confidence = None
        self.data = None
        self.iteration = 0
        self.patch_size = 20
        self.start_time = time.perf_counter()

        # Sécurité pour que l'algorithme se termine :
        # on s'assure qu'on ne fait pas plus de boucles qu'il n'y a de pixels dans l'image
        self.safetyCount = 0
        self.safetyMax = None

    def inpaint(self):
        self.validate_inputs()
        self.initialize_attributes()
        done = False

        while not done:
            self.identify_fill_front()
            self.compute_priorities()

            print('find target...')
            target_patch = self.get_target_patch()
            print('find source...')
            source_pixel, source_patch = self.get_source_patch(target_patch)
            self.update_image(target_patch, source_patch, source_pixel)

            self.update_confidence()
            done = self.is_finished()

        print('total process time : ' + str(time.perf_counter() - self.start_time) + ' seconds')
        return self.image

    def validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def initialize_attributes(self):
        height, width = self.image.shape[:2]
        self.confidence = ((255 - self.mask)/255).astype('uint8')
        self.data = np.zeros([height, width])
        self.safetyMax = height * width
        self.save_temp_image()

    def identify_fill_front(self):
        laplacian = ndimage.laplace(self.mask)
        self.fill_front = (laplacian > 0).astype('uint8')

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

            val = sum(sum(confidence_values)) / (self.patch_size*self.patch_size)
            new_confidence[pixel[0], pixel[1]] = val

        self.confidence = new_confidence

    def update_data(self):
        self.data = 1

    def find_highest_priority(self):
        edge = np.argwhere(self.fill_front > 0)
        max_priority = -np.inf
        max_pixel = None

        for pixel in edge:
            priority = self.priority[pixel[0]][pixel[1]]
            if priority > max_priority:
                max_priority = priority
                max_pixel = pixel

        return max_pixel

    def is_patchable(self, pixel):
        half_size = (self.patch_size - 1) // 2
        height, width = self.image.shape[:2]
        return pixel[0] in [i for i in range(half_size, height-half_size)]\
            and pixel[1] in [j for j in range(half_size, width-half_size)]

    def is_fully_in_mask(self, indices):
        data = self.patch_data(indices, self.fix_mask)
        return data.sum() == 0

    def get_target_patch(self):
        pixel = self.find_highest_priority()
        indices = self.patch_indices(pixel)
        # self.show_patch(self.image, indices, 'target patch')
        return indices

    def get_source_patch(self, target_patch):
        target_data = self.patch_data(target_patch, self.image)
        mask_data = self.patch_data(target_patch, self.mask)
        masked_target = self.mask_data(target_data, mask_data)

        best_match = None
        best_pixel = None
        distance = np.inf
        source_region = np.argwhere(self.fix_mask == 0)

        for pixel in source_region:

            if self.is_patchable(pixel):
                source_patch = self.patch_indices(pixel)

                if self.is_fully_in_mask(source_patch):
                    source_data = self.patch_data(source_patch, self.image)
                    new_distance = self.calculate_distance(masked_target, source_data)

                    if new_distance < distance:
                        distance = new_distance
                        best_pixel = pixel
                        best_match = source_patch

        print('source pixel : '+str(best_pixel))
        # self.show_patch(self.image, best_match, 'source patch')
        return best_pixel, best_match

    def update_image(self, target_patch, source_patch, source_pixel):
        [minX, maxX], [minY, maxY] = target_patch

        target_data = self.patch_data(target_patch, self.image)
        source_data = self.patch_data(source_patch, self.image)
        mask_data = self.patch_data(target_patch, self.mask)

        new_data = self.assemble_data(source_data, target_data, mask_data)

        self.image[minX:maxX, minY:maxY] = new_data
        # self.show_image(self.image, 'updated image')
        self.mask[minX:maxX, minY:maxY] = 0
        self.confidence[minX:maxX, minY:maxY] = self.confidence[source_pixel[0]][source_pixel[1]]

        self.iteration += 1
        self.save_temp_image()

    def is_finished(self):
        white_number = self.mask.sum()
        self.safetyCount += 1
        if white_number == 0 or (self.safetyCount > self.safetyMax):
            return True
        else:
            print(str(self.iteration)+' - remaining : '+str(white_number))
            print('process time : ' + str(time.perf_counter() - self.start_time) + ' seconds'+'\n')
            return False

    def save_temp_image(self):
        imo.imwrite('../Data/Temp/'+str(self.iteration)+'.png', self.image)

    def patch_indices(self, pixel):
        # returns the indices (min and max) of the patch centered around the given pixel

        half_size = (self.patch_size - 1) // 2
        height, width = self.image.shape[:2]

        min_x = max(0, pixel[0] - half_size)
        max_x = min(pixel[0] + half_size, height - 1)
        min_y = max(0, pixel[1] - half_size)
        max_y = min(pixel[1] + half_size, width - 1)

        return [min_x, max_x], [min_y, max_y]

    @staticmethod
    def patch_data(indices, source):
        # returns the pixel values of the given indices in source

        [minX, maxX], [minY, maxY] = indices
        return source[minX:maxX, minY:maxY]

    @staticmethod
    def calculate_distance(target_data, source_data):
        # returns the squared distance between the given target and source patches

        dist = np.sum((target_data-source_data)**2)
        return dist

    @staticmethod
    def mask_data(source, mask, threshold=0):
        # returns the source masked with the given mask [ source * mask ]

        final_patch = np.copy(source)
        height, width = final_patch.shape[:2]

        for i in range(height):
            for j in range(width):
                if mask[i, j] > threshold:
                    final_patch[i][j] = [0, 0, 0]
        return final_patch

    @staticmethod
    def assemble_data(source, target, mask):
        # returns the correct balance between source and target with the given mask

        final_patch = np.copy(target)
        height, width = final_patch.shape[:2]

        for i in range(height):
            for j in range(width):
                source_coeff = mask[i, j]/255
                target_coeff = (255 - mask[i, j])/255
                final_patch[i][j] = final_patch[i][j] * target_coeff + source[i][j] * source_coeff

        return final_patch

    @staticmethod
    def show_image(image, title):
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()

    @staticmethod
    def show_patch(source, indices, title):
        [minX, maxX], [minY, maxY] = indices
        image = np.copy(source)

        image[minX:maxX, minY] = [255, 0, 0]
        image[minX:maxX, maxY] = [255, 0, 0]
        image[minX, minY:maxY] = [255, 0, 0]
        image[maxX, minY:maxY] = [255, 0, 0]

        plt.imshow(image)
        plt.title(title)
        plt.show()




