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

        # Sécurité pour que l'algorithme se termine:
        # on s'assure qu'on ne fait pas plus de boucles qu'il n'y a de pixels dans l'image
        self.safetyCount = 0
        self.safetyMax = None

        #On va travailler sur des copies pour ne pas détruire les images de base 
        self.working_mask = None
        self.working_image = None



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
        self.safetyMax = height * width
        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)

    def identify_fill_front(self):
        laplacian = ndimage.laplace(self.working_mask)
        self.fill_front = (laplacian > 0).astype('uint8')

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
        print(self.working_mask)
        whiteNumber = self.working_mask.sum()
        self.safetyCount += 1 
        #La condition ==0 n'est peut-être pas idéale 
        #En fonction de quand on appelle la fonction, il faudrait peut-être plutôt
        #quelque chose du genre <(nombre de pixels par patch)
        if whiteNumber == 0 or (self.safetyCount < self.safetyMax): 
            return True
        else: 
            return False
        return True #Laisser tant qu'on n'a pas codé le reste 


