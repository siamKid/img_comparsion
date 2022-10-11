import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from skimage import color
import matplotlib.pyplot as plt
import seaborn as sns
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

    def homography(self, img1, img2):
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

    def _get_averaged_images(self, img, kernels):
        return np.array([signal.convolve2d(img, kernel, 'same') for kernel in kernels])
    
    def sort_rows_lexicographically(self, matrix):
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

    def get_histogram(self, image, image_bit_depth=8):
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
        # image_1 = cv2.imread(self.input_path, 1)
        # image_2 = cv2.imread(self.reference_path, 1)
        image_1 = self.input_path
        image_2 = self.reference_path
        
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

class cluster():
    input_path = ""
    n = 0
    window_size = 0
    pca_dim_gray = 0
    pca_dim_rgb = 0
    
    image_1 = 0
    image2_registered = 0
    new_shape = 0
    
    def __init__(self, input_path, n, window_size, pca_dim_gray, pca_dim_rgb):
        self.input_path = input_path
        self.n = n
        self.window_size = window_size
        self.pca_dim_gray = pca_dim_gray
        self.pca_dim_rgb = pca_dim_rgb
    
    def setParam(self, image_1, image2_registered, new_shape):
        self.image_1 = image_1
        self.image2_registered = image2_registered
        self.new_shape = new_shape 
    
    def draw_combination_on_transparent_input_image(self,classes_mse, clustering, combination, transparent_input_image):
        sorted_indexes = np.argsort(classes_mse)
        for class_ in combination:
            c = plt.cm.jet(float(np.argwhere(sorted_indexes == class_))/(len(classes_mse)-1))
            for [i, j] in clustering[class_]:
                transparent_input_image[i, j] = (c[2] * 255, c[1] * 255, c[0] * 255, 255)  #BGR
        return transparent_input_image

    def find_group_of_accepted_classes_DBSCAN(self,MSE_array):
        clustering = DBSCAN(eps=0.02, min_samples=1).fit(np.array(MSE_array).reshape(-1,1))
        number_of_clusters = len(set(clustering.labels_))
        if number_of_clusters == 1:
            print("No significant changes are detected.")
            
        classes = [[] for i in range(number_of_clusters)]
        centers = [0 for i in range(number_of_clusters)]
        for i in range(len(MSE_array)):
            centers[clustering.labels_[i]] += MSE_array[i]
            classes[clustering.labels_[i]].append(i)

        centers = [centers[i]/len(classes[i]) for i in range(number_of_clusters)]
        min_class = centers.index(min(centers))
        accepted_classes = []
        for i in range(len(MSE_array)):
            if clustering.labels_[i] != min_class:
                accepted_classes.append(i)
        
        return [accepted_classes]
    
    def clustering_to_mse_values(self,change_map, img1, img2, n):
        mse =  [0.0 for i in range (0,n)]
        size = [0 for i in range (0,n)]
        img1 = img1.astype(int)
        img2 = img2.astype(int)
        for i in range(change_map.shape[0]):
            for j in range(change_map.shape[1]):
                mse[change_map[i,j]] += np.mean((img1[i,j]-img2[i,j])**2)
                size[change_map[i,j]] += 1
        return [(mse[k]/(255**2))/size[k] for k in range (0,n)], size
    
    def Kmeansclustering(self,FVS, components, images_size):
        kmeans = KMeans(components, verbose=0)
        kmeans.fit(FVS)
        flatten_change_map = kmeans.predict(FVS)
        change_map = np.reshape(flatten_change_map, (images_size[0],images_size[1]))
        return change_map
    
    def find_FVS(self,descriptors, EVS, mean_vec):
        FVS = np.dot(descriptors, EVS)
        FVS = FVS - mean_vec
        return FVS
    
    def find_vector_set(self, descriptors, jump_size):
        descriptors_2d = descriptors.reshape((self.new_shape[0], self.new_shape[1], descriptors.shape[1]))
        vector_set = descriptors_2d[::jump_size,::jump_size]
        vector_set = vector_set.reshape((vector_set.shape[0]*vector_set.shape[1], vector_set.shape[2]))
        mean_vec = np.mean(vector_set, axis=0)
        vector_set = vector_set - mean_vec  # mean normalization
        return vector_set, mean_vec
    
    def descriptors_to_pca(self, descriptors, pca_target_dim, window_size):
        vector_set, mean_vec = self.find_vector_set(descriptors,window_size)
        pca = PCA(pca_target_dim)
        pca.fit(vector_set)
        EVS = pca.components_
        mean_vec = np.dot(mean_vec, EVS.transpose())
        FVS = self.find_FVS(descriptors, EVS.transpose(), mean_vec)
        return FVS
    
    def get_descriptors (self, image1, image2, window_size, pca_dim_gray, pca_dim_rgb):
    
        #################################################   grayscale-diff (abs)

        descriptors = np.zeros((image1.shape[0],image1.shape[1], window_size * window_size))
        diff_image = cv2.absdiff(image1, image2)
        diff_image = color.rgb2gray(diff_image)
        diff_image = np.pad(diff_image,((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)), 'constant')  # default is 0
        
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                descriptors[i,j,:] =diff_image[i:i+window_size,j:j+window_size].ravel()
        descriptors_gray_diff = descriptors.reshape((descriptors.shape[0] * descriptors.shape[1], descriptors.shape[2]))

        #################################################   3-channels-diff (abs)

        descriptors = np.zeros((image1.shape[0], image1.shape[1], window_size * window_size*3))
        diff_image_r = cv2.absdiff(image1[:, :, 0],image2[:, :, 0])
        diff_image_g = cv2.absdiff(image1[:, :, 1],image2[:, :, 1])
        diff_image_b = cv2.absdiff(image1[:, :, 2],image2[:, :, 2])

        diff_image_r = np.pad(diff_image_r, ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
                            'constant')  # default is 0
        diff_image_g = np.pad(diff_image_g,
                                ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
                                'constant')  # default is 0
        diff_image_b = np.pad(diff_image_b,
                                ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
                                'constant')  # default is 0

        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                feature_r = diff_image_r[i:i + window_size, j:j + window_size].ravel()
                feature_g = diff_image_g[i:i + window_size, j:j + window_size].ravel()
                feature_b = diff_image_b[i:i + window_size, j:j + window_size].ravel()
                descriptors[i, j, :] = np.concatenate((feature_r, feature_g, feature_b))
        descriptors_rgb_diff = descriptors.reshape((descriptors.shape[0] * descriptors.shape[1], descriptors.shape[2]))

        #################################################   concatination

        descriptors_gray_diff = self.descriptors_to_pca(descriptors_gray_diff, pca_dim_gray,window_size)
        descriptors_colored_diff = self.descriptors_to_pca(descriptors_rgb_diff, pca_dim_rgb,window_size)

        descriptors = np.concatenate((descriptors_gray_diff, descriptors_colored_diff), axis=1)

        return descriptors
    
    def compute_change_map(self, image1, image2, window_size=5, clusters=16, pca_dim_gray=3, pca_dim_rgb=9):
        descriptors = self.get_descriptors(image1, image2, window_size, pca_dim_gray, pca_dim_rgb)
        # Now we are ready for clustering!
        change_map = self.Kmeansclustering(descriptors, clusters, image1.shape)
        mse_array, size_array = self.clustering_to_mse_values(change_map, image1, image2, clusters)
        sorted_indexes = np.argsort(mse_array)
        colors_array = [plt.cm.jet(float(np.argwhere(sorted_indexes == class_))/(clusters-1)) for class_ in range(clusters)]
        colored_change_map = np.zeros((change_map.shape[0], change_map.shape[1], 3), np.uint8)
        palette_colored_change_map = np.zeros((change_map.shape[0], change_map.shape[1], 3), np.uint8)
        palette = sns.color_palette("Paired", clusters)
        for i in range(change_map.shape[0]):
            for j in range(change_map.shape[1]):
                colored_change_map[i, j]= (255*colors_array[change_map[i,j]][0],255*colors_array[change_map[i,j]][1],255*colors_array[change_map[i,j]][2])
                palette_colored_change_map[i, j] = [255*palette[change_map[i, j]][0],255*palette[change_map[i, j]][1],255*palette[change_map[i, j]][2]]

        return change_map, mse_array, size_array
    
    def clustering(self):
        clustering_map, mse_array, size_array = self.compute_change_map(
            self.image_1, 
            self.image2_registered, 
            window_size=self.window_size,
            clusters=self.n, 
            pca_dim_gray= self.pca_dim_gray, 
            pca_dim_rgb=self.pca_dim_rgb)

        clustering = [[] for _ in range(n)]
        for i in range(clustering_map.shape[0]):
            for j in range(clustering_map.shape[1]):
                clustering[int(clustering_map[i,j])].append([i,j])

        #input_image = cv2.imread(self.input_path)
        input_image = self.input_path
        input_image = cv2.resize(input_image, self.new_shape, interpolation=cv2.INTER_AREA)
        b_channel, g_channel, r_channel = cv2.split(input_image)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        alpha_channel[:, :] = 50
        groups = self.find_group_of_accepted_classes_DBSCAN(mse_array)
        for group in groups:
            transparent_input_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
            result = self.draw_combination_on_transparent_input_image(mse_array, clustering, group, transparent_input_image)
            
        return result
        
    
def main(input_path,reference_path,n,window_size, pca_dim_gray, pca_dim_rgb,
          lighting_fix, use_homography, resize_factor):
    
    processing = pre_processing(input_path=input_path,
                  reference_path=reference_path,
                  resize_factor=resize_factor, 
                  use_homography=use_homography,
                  lighting_fix=lighting_fix)
    image_1, image2_registered, new_shape = processing.processing()
    
    clust = cluster(input_path=input_path,
                  n=n,
                  window_size=window_size, 
                  pca_dim_gray=pca_dim_gray,
                  pca_dim_rgb=pca_dim_rgb)
    clust.setParam(image_1,image2_registered,new_shape)
    
    return clust.clustering()
    

    
if __name__ == '__main__':
    with open('pre_processing.pkl','wb') as f:
        dill.dump(pre_processing,f)
        
    with open('cluster.pkl','wb') as f:
        dill.dump(cluster,f)
        
        
    
    
    # input_path = "img/golden.jpg"
    # reference_path = "img/diff1.jpg"
    # n=15
    # window_size = 5
    # pca_dim_gray = 3
    # pca_dim_rgb = 9
    
    # lighting_fix = True
    # use_homography = True
    # resize_factor = float(1)
    
    
    # result = main(input_path, reference_path, int(n), int(window_size),
    #      int(pca_dim_gray), int(pca_dim_rgb), bool(lighting_fix), bool(use_homography),
    #      float(resize_factor))
    
    # cv2.imwrite('ACCEPTED_CLASSES1'+'.png', result)
    