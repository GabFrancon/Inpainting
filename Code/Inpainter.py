import numpy as np
import imageio as imo
import matplotlib.pyplot as plt
from scipy import ndimage


class Inpainter:
    def __init__(self, image, mask):
        self.image = np.copy(image.astype('uint8'))
        self.mask = np.copy(mask.round().astype('uint8'))
        self.fill_front = None
        self.priority = None
        self.confidence = None
        self.data = None
        self.patch_size = 9

    def inpaint(self):
        self.validate_inputs()
        self.initialize_attributes()
        done = False

        while not done:
            self.identify_fill_front()
            self.compute_priorities()

            target_patch = self.get_target_patch()
            source_patch = self.get_source_patch(target_patch)

            self.update_image(target_patch, source_patch)

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

    def identify_fill_front(self):
        laplacian = ndimage.laplace(self.mask)
        self.fill_front = (laplacian > 0).astype('uint8')
        print(self.fill_front)

    def compute_priorities(self):
        self.update_confidence()
        self.update_data()
        #self.priority = self.confidence * self.data * self.fill_front

    def get_target_patch(self):
        print('get target patch\n')
        return 0

    def get_source_patch(self, target_patch):
        print('get source patch\n')
        return 0

    def update_image(self, target_patch, source_patch):
        print('update image\n')

    def update_confidence(self):
        print('update confidence\n')

    def update_data(self):
        print('update data\n')

    def is_finished(self):
        print('test if process finished\n')
        return True



