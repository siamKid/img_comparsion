import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from skimage import color
import matplotlib.pyplot as plt
import seaborn as sns

import dill

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
    
    @staticmethod
    def draw_combination_on_transparent_input_image(classes_mse, clustering, combination, transparent_input_image):
        sorted_indexes = np.argsort(classes_mse)
        for class_ in combination:
            c = plt.cm.jet(float(np.argwhere(sorted_indexes == class_))/(len(classes_mse)-1))
            for [i, j] in clustering[class_]:
                transparent_input_image[i, j] = (c[2] * 255, c[1] * 255, c[0] * 255, 255)  #BGR
        return transparent_input_image
    
    @staticmethod
    def find_group_of_accepted_classes_DBSCAN(MSE_array):
        clustering = DBSCAN(eps=0.02, min_samples=1).fit(np.array(MSE_array).reshape(-1,1))
        number_of_clusters = len(set(clustering.labels_))
        if number_of_clusters == 1:
            print("No significant changes are detected.")
            exit(0)
        
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
    
    @staticmethod
    def clustering_to_mse_values(change_map, img1, img2, n):
        mse =  [0.0 for i in range (0,n)]
        size = [0 for i in range (0,n)]
        img1 = img1.astype(int)
        img2 = img2.astype(int)
        for i in range(change_map.shape[0]):
            for j in range(change_map.shape[1]):
                mse[change_map[i,j]] += np.mean((img1[i,j]-img2[i,j])**2)
                size[change_map[i,j]] += 1
        return [(mse[k]/(255**2))/size[k] for k in range (0,n)], size
    
    @staticmethod
    def Kmeansclustering(FVS, components, images_size):
        kmeans = KMeans(components, verbose=0)
        kmeans.fit(FVS)
        flatten_change_map = kmeans.predict(FVS)
        change_map = np.reshape(flatten_change_map, (images_size[0],images_size[1]))
        return change_map
    
    @staticmethod
    def find_FVS(descriptors, EVS, mean_vec):
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

        input_image = cv2.imread(self.input_path)
        input_image = cv2.resize(input_image, self.new_shape, interpolation=cv2.INTER_AREA)
        b_channel, g_channel, r_channel = cv2.split(input_image)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        alpha_channel[:, :] = 50
        groups = self.find_group_of_accepted_classes_DBSCAN(mse_array)
        for group in groups:
            transparent_input_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
            result = self.draw_combination_on_transparent_input_image(mse_array, clustering, group, transparent_input_image)
            
        return result
        
        
if __name__ == "__main__":
    with open('cluster.pkl','wb') as f:
        dill.dump(cluster,f)