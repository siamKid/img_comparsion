import numpy as np
import cv2
from scipy import signal

import dill

class pre_processing():
    _kernel1 = 1.0 / 5.0 * np.array([[0, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 0]])

    _kernel2 = 1.0 / 9.0 * np.array([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 1]])

    _kernel3 = 1.0 / 13.0 * np.array([[0, 0, 1, 0, 0],
                                      [0, 1, 1, 1, 0],
                                      [1, 1, 1, 1, 1],
                                      [0, 1, 1, 1, 0],
                                      [0, 0, 1, 0, 0]])

    _kernel4 = 1.0 / 21.0 * np.array([[0, 1, 1, 1, 0],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [0, 1, 1, 1, 0]])

    _kernel5 = 1.0 / 25.0 * np.array([[1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1]])
    
    _kernel_mapping = {1: [_kernel1],
                       2: [_kernel1, _kernel2],
                       3: [_kernel1, _kernel2, _kernel3],
                       4: [_kernel1, _kernel2, _kernel3, _kernel4],
                       5: [_kernel1, _kernel2, _kernel3, _kernel4, _kernel5]}
    
    input_path = ""
    reference_path = ""
    
    use_homography = False
    lighting_fix = False
    
    resize_factor = 1.0
    
    size_0 = 0
    size_1 = 0
    

    def __init__(self, input_path="", reference_path="", resize_factor=1.0, use_homography=False, lighting_fix=False):
        self.input_path = input_path
        self.reference_path = reference_path
        self.resize_factor = float(resize_factor)
        self.use_homography = use_homography
        self.lighting_fix = lighting_fix
        

    def set_size(self, size_00, size_11):
        self.size_0 = size_00
        self.size_1 = size_11

    @staticmethod
    def homography(img1, img2):
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des2, des1, k=2)

        # Apply ratio test
        good_draw = []
        good_without_list = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance: #0.8 = a value suggested by David G. Lowe.
                good_draw.append([m])
                good_without_list.append(m)

        # Extract location of good matches
        points1 = np.zeros((len(good_without_list), 2), dtype=np.float32)
        points2 = np.zeros((len(good_without_list), 2), dtype=np.float32)

        for i, match in enumerate(good_without_list):
            points1[i, :] = kp2[match.queryIdx].pt
            points2[i, :] = kp1[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width = img2.shape[:2]
        white_img2 = 255- np.zeros(shape=img2.shape, dtype=np.uint8)
        whiteReg = cv2.warpPerspective(white_img2, h, (width, height))
        blank_pixels_mask = np.any(whiteReg != [255, 255, 255], axis=-1)
        im2Reg = cv2.warpPerspective(img2, h, (width, height))

        return im2Reg, blank_pixels_mask
    
    @staticmethod
    def _get_averaged_images(img, kernels):
        return np.array([signal.convolve2d(img, kernel, 'same') for kernel in kernels])
    
    @staticmethod
    def sort_rows_lexicographically(matrix):
        rotated_matrix = np.rot90(matrix)

        sorted_indices = np.lexsort(rotated_matrix)
        return matrix[sorted_indices]
    
    def _get_average_values_for_every_pixel(self, img, number_kernels):
        kernels = self._kernel_mapping[number_kernels]
        averaged_images = self._get_averaged_images(img, kernels)
        img_size = averaged_images[0].shape[0] * averaged_images[0].shape[1]

        reshaped_averaged_images = averaged_images.reshape((number_kernels, img_size))
        transposed_averaged_images = reshaped_averaged_images.transpose()
        return transposed_averaged_images
    
    def _match_to_histogram(self, image, reference_histogram, number_kernels):
        img_size = image.shape[0] * image.shape[1]

        merged_images = np.empty((img_size, number_kernels + 2))

        merged_images[:, 0] = image.reshape((img_size,))

        indices_of_flattened_image = np.arange(img_size).transpose()
        merged_images[:, -1] = indices_of_flattened_image

        averaged_images = self._get_average_values_for_every_pixel(image, number_kernels)
        for dimension in range(0, number_kernels):
            merged_images[:, dimension + 1] = averaged_images[:, dimension]

        sorted_merged_images = self.sort_rows_lexicographically(merged_images)

        index_start = 0
        for gray_value in range(0, len(reference_histogram)):
            index_end = int(index_start + reference_histogram[gray_value])
            sorted_merged_images[index_start:index_end, 0] = gray_value
            index_start = index_end

        sorted_merged_images = sorted_merged_images[sorted_merged_images[:, -1].argsort()]
        new_target_img = sorted_merged_images[:, 0].reshape(image.shape)

        return new_target_img
    
    def match_image_to_histogram(self, image, reference_histogram, number_kernels=5):
        if len(image.shape) == 3:
            # Image with more than one dimension. I. e. an RGB image.
            output = np.empty(image.shape)
            dimensions = image.shape[2]

            for dimension in range(0, dimensions):
                output[:, :, dimension] = self._match_to_histogram(image[:, :, dimension],reference_histogram[:, dimension], number_kernels)
        else:
            # Gray value image
            output = self._match_to_histogram(image, reference_histogram, number_kernels)

        return output
    
    @staticmethod
    def get_histogram(image, image_bit_depth=8):
        max_grey_value = pow(2, image_bit_depth)

        if len(image.shape) == 3:
            dimensions = image.shape[2]
            hist = np.empty((max_grey_value, dimensions))

            for dimension in range(0, dimensions):
                for gray_value in range(0, max_grey_value):
                    image_2d = image[:, :, dimension]
                    hist[gray_value, dimension] = len(image_2d[image_2d == gray_value])
        else:
            hist = np.empty((max_grey_value,))

            for gray_value in range(0, max_grey_value):
                hist[gray_value] = len(image[image == gray_value])

        return hist
    
    def light_diff_elimination(self, image1, image2_registered):
        reference_histogram = self.get_histogram(image1)
        new_target_img = self.match_image_to_histogram(image2_registered, reference_histogram)
        new_target_img = np.asarray(new_target_img, dtype=np.uint8)
        return new_target_img
    
    def processing(self):
        #read the inputs
        image_1 = cv2.imread(self.input_path, 1)
        image_2 = cv2.imread(self.reference_path, 1)
        
        #we need the images to be the same size. resize_factor is for increasing or decreasing further the images
        new_shape = (int(self.resize_factor*0.5*(image_1.shape[1]+image_2.shape[1])), int(self.resize_factor*0.5*(image_1.shape[0]+image_2.shape[0])))
        image_1 = cv2.resize(image_1, new_shape, interpolation=cv2.INTER_AREA)
        image_2 = cv2.resize(image_2, new_shape, interpolation=cv2.INTER_AREA)
        self.set_size(new_shape[0],new_shape[1])
        
        if self.use_homography:
            image2_registered, blank_pixels = self.homography(image_1, image_2)
        else:
            image2_registered  = image_2
            
        if self.use_homography:
            image_1[blank_pixels] = [0,0,0]
            image2_registered[blank_pixels] = [0, 0, 0]
        
        if (self.lighting_fix):
            image2_registered = self.light_diff_elimination(image_1, image2_registered)
        
        return image_1, image2_registered, new_shape
    

if __name__=="__main__":
    with open('pre_processing.pkl','wb') as f:
        dill.dump(pre_processing,f)
        
    