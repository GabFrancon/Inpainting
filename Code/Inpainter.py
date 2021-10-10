import time
import numpy as np
import matplotlib.pyplot as plt
import imageio as imo
from scipy import ndimage


class Inpainter:
    def __init__(self, image, mask):
        self.mask = np.copy(mask.astype('uint8'))
        self.fix_mask = np.copy(self.mask)
        self.image = np.copy(image.astype('uint8'))
        self.fill_front = None
        self.priority = None
        self.confidence = None
        self.data = None
        self.iteration = 0
        self.patch_size = 9
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

            target_pixel, target_patch, target_data, local_mask = self.get_target()
            best_source = self.get_source(target_patch, target_data, local_mask)

            self.update_image(target_patch, target_data+best_source, target_pixel)
            done = self.is_finished()

        print('total process time : ' + str(time.perf_counter() - self.start_time) + ' seconds')
        return self.image

    def validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def initialize_attributes(self):
        height, width = self.image.shape[:2]
        self.confidence = ((255 - self.mask)/255)
        self.data = np.zeros([height, width])
        self.safetyMax = height * width
        self.mask_input()
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

    def get_target(self):
        pixel = self.find_highest_priority()
        indices = self.patch_indices(pixel)
        data = self.patch_data(indices, self.image)
        local_mask = self.get_local_mask(indices)
        target = self.mask_data(data, local_mask)
        # self.show_patch(self.image, indices, 'target patch')
        # self.show_image(target, 'target')
        return pixel, indices, target, local_mask

    def get_local_mask(self, indices):
        return self.patch_data(indices, self.mask)

    def get_source(self, target_patch, masked_target, local_mask):
        best_pixel = None
        best_match = None
        distance = np.inf
        source_region = np.argwhere(self.fix_mask == 0)

        for pixel in source_region:

            if self.is_patchable(pixel):
                source_patch = self.patch_indices(pixel)

                if self.is_fully_in_mask(source_patch):
                    source_data = self.patch_data(source_patch, self.image)
                    masked_source = self.mask_data(source_data, local_mask)
                    new_distance = self.calculate_distance(target_patch, source_patch, masked_target, masked_source)

                    if new_distance < distance:
                        distance = new_distance
                        best_pixel = pixel
                        best_match = source_data

        best_source = self.mask_data(best_match, 255 - local_mask)
        print('source pixel : '+str(best_pixel))
        # self.show_patch(self.image, best_location, 'source patch')
        # self.show_image(best_source, 'source')
        return best_source

    def update_image(self, location, data, source_pixel):
        [minX, maxX], [minY, maxY] = location

        # self.show_image(data, 'new data')
        self.image[minX:maxX, minY:maxY] = data
        self.mask[minX:maxX, minY:maxY] = 0
        self.confidence[minX:maxX, minY:maxY] = self.confidence[source_pixel[0]][source_pixel[1]]

        self.iteration += 1
        self.save_temp_image()
        # self.show_image(self.image, 'updated image')

    def is_finished(self):
        white_number = self.mask.sum()
        self.safetyCount += 1
        if white_number == 0 or (self.safetyCount > self.safetyMax):
            return True
        else:
            print(str(self.iteration)+' - remaining : '+str(white_number))
            print('process time : ' + str(time.perf_counter() - self.start_time) + ' seconds'+'\n')
            return False

    def mask_input(self):
        mask_pos = np.argwhere(self.mask > 128)

        # color inside the mask area of the input in white
        for p in mask_pos:
            self.image[p[0], p[1]] = 255

        # color the outlines in red
        '''self.identify_fill_front()
        front_pos = np.argwhere(self.fill_front == 1)

        for p in front_pos:
            self.image[p[0], p[1]] = (255, 0, 0)'''

    def save_temp_image(self):
        if self.iteration % 5 == 0:
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
    def calculate_distance(target_patch, source_patch, target_data, source_data):
        # returns the squared distance between the given target and source patches

        [minXT, maxXT], [minYT, maxYT] = target_patch
        [minXS, maxXS], [minYS, maxYS] = source_patch

        euclidian_dist = np.sqrt((minXT - minXS)**2 + (minYT - minYS)**2)
        squared_dist = ((target_data-source_data)**2).sum()

        return squared_dist + euclidian_dist

    @staticmethod
    def mask_data(source, mask, threshold=128):
        # returns the source masked with the given mask [ source * mask ]

        final_patch = np.copy(source)
        height, width = final_patch.shape[:2]

        for i in range(height):
            for j in range(width):
                if mask[i, j] > threshold:
                    final_patch[i][j] = 0
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




