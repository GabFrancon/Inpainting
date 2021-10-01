class Inpainter():
    def __init__(self, image, mask):
        self.image = image.astype('uint8')
        self.mask = mask.round().astype('uint8')

    def inpaint(self):
        if self.image.shape != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')
        else:
            return self.image

