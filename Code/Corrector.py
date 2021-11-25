import imageio as imo
from skimage.color import rgb2hsv, rgb2gray
from utils import *


class Corrector:
    def __init__(self, image, mask, shift_map):
        self.image     = np.copy(image).astype('uint8')
        self.mask      = np.copy(mask).astype('uint8')
        self.texture   = None
        self.shift_map = np.copy(shift_map).astype('uint8')

        self.patch_size = 7
        self.real_mask = None
        self.k_max     = 5
        self.r_max     = np.max(self.mask.shape)
        self.rho       = 0.5

    def initialize_attribute(self):
        self.real_mask = self.delimit_real_mask()
        self.texture   = self.initialize_texture()

    def correct(self):
        print('\n\ncorrection of previous inpainting...')
        print_progress(0, self.k_max)
        self.save_temp_image('0')
        start_time = time.perf_counter()
        self.initialize_attribute()
        k, e = 0, 1

        while e > 0.1 and k < self.k_max:
            img = np.copy(self.image).astype('uint8')
            self.ann_search()
            print_progress(k + 0.33, self.k_max)
            self.image = self.reconstruction(self.image)
            print_progress(k + 0.66, self.k_max)
            self.texture = self.reconstruction(self.texture)
            print_progress(k + 1, self.k_max)

            e = np.linalg.norm(img - self.image) / (3 * np.sum(self.real_mask))
            k += 1
            self.save_temp_image(str(k))

        print('\ntotal process time : ' + get_chrono(start_time))
        print('total iteration    : ' + str(k)+'\n')

        return self.image

    def delimit_real_mask(self):
        real_mask = np.copy(self.mask)
        source = np.argwhere(self.mask == 0)
        for p in source:
            if self.average_patch(p, self.mask) > 0:
                real_mask[p[0], p[1]] = 1
        return real_mask

    def ann_search(self):
        hole = np.argwhere(self.mask == 1)
        rev_hole = np.flip(hole, axis=0)

        for k in range(self.k_max):
            if k < self.k_max//2:
                for pixel in hole:
                    p = np.copy(pixel)
                    args = np.array([p - np.array([0, 1]), p - np.array([1, 0])])
                    self.ann_iter(p, args)
            else:
                for pixel in rev_hole:
                    p = np.copy(pixel)
                    args = np.array([p + np.array([0, 1]), p + np.array([1, 0])])
                    self.ann_iter(p, args)

    def ann_iter(self, p, args):
        p_nn = p + self.shift(p)
        if not self.is_patchable(p_nn):
            return

        h, w = self.mask.shape
        nn_list = [p_nn]
        if 0 <= args[0][0] < h and 0 <= args[0][1] < w:
            a_nn = p + self.shift(args[0])
            if self.is_patchable(a_nn):
                nn_list.append(a_nn)
        if 0 <= args[1][0] < h and 0 <= args[1][1] < w:
            b_nn = p + self.shift(args[1])
            if self.is_patchable(b_nn):
                nn_list.append(b_nn)

        dist_list = [self.calculate_distance(p, nn) for nn in nn_list]
        q = nn_list[np.argmin(np.array(dist_list))]
        self.shift_map[p[0], p[1]] = q - p

        z_max = int(- np.log(np.max(self.mask.shape)) // np.log(self.rho))
        for z in range(z_max):
            random_val = np.around(self.r_max * self.rho ** z * np.random.uniform(low=-1, high=1, size=2)).astype('int')
            q = p + self.shift_map[p[0], p[1]] + random_val

            if self.is_patchable(q):
                if self.calculate_distance(p, q) < self.calculate_distance(p, p_nn):
                    self.shift_map[p[0], p[1]] = q - p

    def reconstruction(self, img):
        hole = np.argwhere(self.mask == 1)
        new_img = np.copy(img)
        h, w = self.mask.shape
        for p in hole:
            patch = self.patch_indices(p)
            val = np.zeros_like(patch_data(patch, img))
            count = 0
            for x in range(val.shape[0]):
                for y in range(val.shape[1]):
                    nn = p + self.shift([patch[0][0]+x, patch[1][0]+y])
                    if 0 <= nn[0] < h and 0 <= nn[1] < w:
                        val[x, y] = img[nn[0], nn[1]]
                        count += 1

            if count != 0:
                new_img[p[0], p[1]] = np.sum(np.sum(val, axis=0), axis=0)/count

        return new_img

    def patch_indices(self, pixel):
        # returns the indices (min and max) of the patch centered around the given pixel
        half_size = (self.patch_size - 1) // 2
        height, width = self.image.shape[:2]

        min_x = max(0, pixel[0] - half_size)
        max_x = min(pixel[0] + half_size, height - 1)
        min_y = max(0, pixel[1] - half_size)
        max_y = min(pixel[1] + half_size, width - 1)

        return [min_x, max_x + 1], [min_y, max_y + 1]

    def get_local_mask(self, indices):
        return patch_data(indices, np.copy(self.real_mask))

    def save_temp_image(self, name):
        # saves the result of the last iteration
        imo.imwrite('../Data/Temp_corr/' + name + '.jpg', self.image)

    def calculate_distance(self, target, source):
        target_patch = self.patch_indices(target)
        source_patch = self.patch_indices(source)
        target_color = rgb2hsv(patch_data(target_patch, self.image))
        source_color = rgb2hsv(patch_data(source_patch, self.image))
        target_texture = patch_data(target_patch, self.texture)
        source_texture = patch_data(source_patch, self.texture)

        color_dist = (source_color - target_color) ** 2
        texture_dist = (source_texture - target_texture) ** 2
        return color_dist.sum() + 50 * texture_dist.sum()

    def initialize_texture(self):
        h, w = self.mask.shape
        texture = np.zeros((h, w, 2))
        derivX, derivY = np.array(np.gradient(rgb2gray(self.image)))
        deriv = np.dstack((derivX, derivY))

        for x in range(h):
            for y in range(w):
                texture[x, y] = self.average_patch(np.array([x, y]), deriv)

        return texture

    def average_patch(self, pixel, source):
        patch = self.patch_indices(pixel)
        data = patch_data(patch, source)
        return sum(sum(data)) / (self.patch_size * self.patch_size)

    def shift(self, p):
        return self.shift_map[p[0], p[1]]

    def is_patchable(self, p):
        h, w = self.mask.shape
        half = self.patch_size // 2
        bool1 = (half < p[0] < h-half) and (half < p[1] < w-half)
        patch = self.patch_indices(p)
        data = patch_data(patch, self.real_mask)
        bool2 = (data.sum() == 0)
        return bool1 and bool2
