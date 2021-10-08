import numpy as np
import matplotlib.pyplot as plt
import imageio as imo
from scipy import ndimage


class Inpainter:
    def __init__(self, image, mask):
        self.mask = np.copy(mask.astype('uint8'))
        self.image = self.rgba2rgb(np.copy(image.astype('uint8')))
        self.fill_front = None
        self.priority = None
        self.confidence = None
        self.data = None
        self.iteration = 0
        self.patch_size = 20

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

        return self.image

    def validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def initialize_attributes(self):
        height, width = self.image.shape[:2]
        self.confidence = (1 - self.mask).astype(float)
        self.data = np.zeros([height, width])
        self.safetyMax = height * width

    def identify_fill_front(self):
        laplacian = ndimage.laplace(self.mask)
        self.fill_front = (laplacian > 0).astype('uint8')

    def compute_priorities(self):
        self.update_confidence()
        self.update_data()
        self.priority = self.confidence * self.data

    def get_target_patch(self):
        pixel = self.find_highest_priority()
        indices = self.patch_indices(pixel)
        #self.show_patch(indices)
        return indices

    def get_source_patch(self, target_patch):
        target_data = self.patch_data(target_patch, self.image)
        mask_data = self.patch_data(target_patch, self.mask)

        masked_target = self.masking_data(target_data, mask_data)

        best_match = None
        best_pixel = None
        distance = float('inf')
        source_region = self.get_source_region()

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
        #self.show_patch(best_match)
        return best_pixel, best_match

    def update_image(self, target_patch, source_patch, source_pixel):
        [minX, maxX], [minY, maxY] = target_patch
        target_data = self.patch_data(target_patch, self.image)
        source_data = self.patch_data(source_patch, self.image)
        mask_data = self.patch_data(target_patch, self.mask)

        new_data = self.assemble_data(target_data, source_data, mask_data)

        self.image[minX:maxX, minY:maxY] = new_data
        self.mask[minX:maxX, minY:maxY] = 0
        self.confidence[minX:maxX, minY:maxY] = self.confidence[source_pixel[0]][source_pixel[1]]

        self.iteration += 1
        self.save_temp_image()

    def update_confidence(self):
        new_confidence = np.copy(self.confidence)
        edge = np.argwhere(self.fill_front > 0)

        for pixel in edge:
            indices = self.patch_indices(pixel)
            confidence_values = self.patch_data(indices, self.confidence)

            val = sum(sum(confidence_values)) / (self.patch_size*self.patch_size)
            new_confidence[pixel[0], pixel[1]] = val

        self.confidence = new_confidence

    def get_source_region(self):
        source = np.argwhere(self.mask == 0)
        return source

    def is_patchable(self, pixel):
        half_size = (self.patch_size - 1) // 2
        height, width = self.image.shape[:2]
        return pixel[0] in [i for i in range(half_size, height-half_size)]\
            and pixel[1] in [j for j in range(half_size, width-half_size)]

    def is_fully_in_mask(self, indices):
        data = self.patch_data(indices, self.mask)
        return data.sum() == 0

    def update_data(self):
        self.data = 1

    def find_highest_priority(self):
        pixel = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return pixel

    @staticmethod
    def rgba2rgb(rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        if ch == 3:
            return rgba

        assert ch == 4, 'RGBA image has 4 channels.'

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

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
        dist = np.sum((target_data-source_data)**2)
        return dist

    @staticmethod
    def assemble_data(target_data, source_data, mask_data):
        final_patch = target_data
        height, width = final_patch.shape[:2]

        for i in range(height):
            for j in range(width):
                if mask_data[i, j] > 0:
                    final_patch[i][j] = source_data[i][j]

        return final_patch

    @staticmethod
    def masking_data(target_data, mask_data):
        final_patch = np.copy(target_data)
        height, width = final_patch.shape[:2]

        for i in range(height):
            for j in range(width):
                if mask_data[i, j] > 0:
                    final_patch[i][j] = [0, 0, 0]

        return final_patch

    def is_finished(self):
        white_number = self.mask.sum()
        self.safetyCount += 1 
        #La condition ==0 n'est peut-être pas idéale 
        #En fonction de quand on appelle la fonction, il faudrait peut-être plutôt
        #quelque chose du genre <(nombre de pixels par patch)
        if white_number == 0 or (self.safetyCount > self.safetyMax):
            return True
        else:
            print(str(self.iteration)+' - remaining : '+str(white_number)+'\n')
            return False

    @staticmethod
    def show_image(image):
        plt.imshow(image, cmap='gray')
        plt.show()

    def show_patch(self, indices):
        [minX, maxX], [minY, maxY] = indices
        image = np.copy(self.image)

        image[minX:maxX, minY] = [255, 0, 0]
        image[minX:maxX, maxY] = [255, 0, 0]
        image[minX, minY:maxY] = [255, 0, 0]
        image[maxX, minY:maxY] = [255, 0, 0]

        plt.imshow(image)
        plt.show()

    def save_temp_image(self):
        imo.imwrite('../Data/Temp/'+str(self.iteration)+'.png', self.image)





