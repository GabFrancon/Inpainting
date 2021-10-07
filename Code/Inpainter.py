import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


class Inpainter:
    def __init__(self, image, mask):
        self.mask = np.copy(mask.astype('uint8'))
        self.image = np.copy(image.astype('uint8'))
        self.fill_front = None
        self.priority = None
        self.confidence = None
        self.data = None
        self.patch_size = 15

        # Sécurité pour que l'algorithme se termine:
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

            pixel = self.fill_front[0]
            target_patch = self.get_target_patch(pixel)
            source_patch = self.get_source_patch(target_patch)
            self.update_image(target_patch, source_patch)

            #self.show_image()

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
        self.fill_front = np.argwhere(self.fill_front > 0)

    def compute_priorities(self):
        self.update_confidence()
        self.update_data()
        #self.priority = self.confidence * self.data * self.fill_front

    def get_target_patch(self, pixel):
        return self.get_patch_indices(pixel)

    def get_source_patch(self, target_patch):
        x, y = target_patch
        pixel = x[0], y[0]
        [min_x, max_x], [min_y, max_y] = self.get_patch_indices(pixel)
        source_patch = self.image[min_x:max_x, min_y:max_y]

        return source_patch

    def update_image(self, target_patch, source_patch):
        [min_x, max_x], [min_y, max_y] = target_patch
        self.image[min_x:max_x, min_y:max_y] = source_patch
        self.mask[min_x:max_x, min_y:max_y] = 0

    def update_confidence(self):
        self.confidence = self.confidence

    def update_data(self):
        self.data = self.data

    def get_patch_indices(self, pixel):
        # return index of the patch limits in both x and y axis

        half_size = (self.patch_size - 1) // 2
        height, width = self.image.shape[:2]

        min_x = max(0, pixel[0] - half_size)
        max_x = min(pixel[0] + half_size, height - 1)
        min_y = max(0, pixel[1] - half_size)
        max_y = min(pixel[1] + half_size, width - 1)

        indices = [min_x, max_x], [min_y, max_y]
        return indices

    def is_finished(self):
        white_number = self.mask.sum()
        self.safetyCount += 1 
        #La condition ==0 n'est peut-être pas idéale 
        #En fonction de quand on appelle la fonction, il faudrait peut-être plutôt
        #quelque chose du genre <(nombre de pixels par patch)
        if white_number == 0 or (self.safetyCount > self.safetyMax):
            return True
        else: 
            return False

    def show_image(self):
        plt.imshow(self.image, cmap='gray')
        plt.show()


