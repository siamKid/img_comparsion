import numpy as np
from scipy import signal
import numpy as np
import cv2
import global_variables

class ExactHistogramMatcher:
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

    @staticmethod
    def _get_averaged_images(img, kernels):
        return np.array([signal.convolve2d(img, kernel, 'same') for kernel in kernels])

    @staticmethod
    def _get_average_values_for_every_pixel(img, number_kernels):
        kernels = ExactHistogramMatcher._kernel_mapping[number_kernels]
        averaged_images = ExactHistogramMatcher._get_averaged_images(img, kernels)
        img_size = averaged_images[0].shape[0] * averaged_images[0].shape[1]

        reshaped_averaged_images = averaged_images.reshape((number_kernels, img_size))
        transposed_averaged_images = reshaped_averaged_images.transpose()
        return transposed_averaged_images

    @staticmethod
    def sort_rows_lexicographically(matrix):
        rotated_matrix = np.rot90(matrix)

        sorted_indices = np.lexsort(rotated_matrix)
        return matrix[sorted_indices]

    @staticmethod
    def _match_to_histogram(image, reference_histogram, number_kernels):
        img_size = image.shape[0] * image.shape[1]

        merged_images = np.empty((img_size, number_kernels + 2))

        merged_images[:, 0] = image.reshape((img_size,))

        indices_of_flattened_image = np.arange(img_size).transpose()
        merged_images[:, -1] = indices_of_flattened_image

        averaged_images = ExactHistogramMatcher._get_average_values_for_every_pixel(image, number_kernels)
        for dimension in range(0, number_kernels):
            merged_images[:, dimension + 1] = averaged_images[:, dimension]

        sorted_merged_images = ExactHistogramMatcher.sort_rows_lexicographically(merged_images)

        index_start = 0
        for gray_value in range(0, len(reference_histogram)):
            index_end = int(index_start + reference_histogram[gray_value])
            sorted_merged_images[index_start:index_end, 0] = gray_value
            index_start = index_end

        sorted_merged_images = sorted_merged_images[sorted_merged_images[:, -1].argsort()]
        new_target_img = sorted_merged_images[:, 0].reshape(image.shape)

        return new_target_img

    @staticmethod
    def match_image_to_histogram(image, reference_histogram, number_kernels=5):
        if len(image.shape) == 3:
            # Image with more than one dimension. I. e. an RGB image.
            output = np.empty(image.shape)
            dimensions = image.shape[2]

            for dimension in range(0, dimensions):
                output[:, :, dimension] = ExactHistogramMatcher._match_to_histogram(image[:, :, dimension],
                                                                                    reference_histogram[:, dimension],
                                                                                    number_kernels)
        else:
            # Gray value image
            output = ExactHistogramMatcher._match_to_histogram(image,
                                                               reference_histogram,
                                                               number_kernels)

        return output


def light_diff_elimination_NAIVE(image1, image2_registered):
    img_hsv1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    img_cpy1 = np.copy(img_hsv1)
    img_hsv2 = cv2.cvtColor(image2_registered, cv2.COLOR_RGB2HSV)
    img_cpy2 = np.copy(img_hsv2)
    for i in range(img_hsv1.shape[0]):
        for j in range(img_hsv1.shape[1]):
            if img_cpy1[i, j, 1] >= 50 or img_cpy1[i, j, 2] <= 205:
                img_cpy1[i, j, 1] = (img_hsv1[i, j, 1] + img_hsv2[i, j, 1])
    for i in range(img_hsv1.shape[0]):
        for j in range(img_hsv1.shape[1]):
            if img_cpy2[i, j, 1] >= 50 or img_cpy2[i, j, 2] <= 205:
                img_cpy2[i, j, 1] = (img_hsv1[i, j, 1] + img_hsv2[i, j, 1])
                
    image1 = cv2.cvtColor(img_cpy1, cv2.COLOR_HSV2RGB)
    image2_registered = cv2.cvtColor(img_cpy2, cv2.COLOR_HSV2RGB)
    
    return image1, image2_registered

#rgb - are the images in rgb colors of just gray?
def light_diff_elimination(image1, image2_registered):
    reference_histogram = ExactHistogramMatcher.get_histogram(image1)
    new_target_img = ExactHistogramMatcher.match_image_to_histogram(image2_registered, reference_histogram)
    new_target_img = np.asarray(new_target_img, dtype=np.uint8)
    return new_target_img