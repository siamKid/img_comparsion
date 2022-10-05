import numpy as np
import cv2
from PCA_Kmeans import compute_change_map, find_group_of_accepted_classes_DBSCAN, draw_combination_on_transparent_input_image
import global_variables

def main(input_path,reference_path,n,window_size, pca_dim_gray, pca_dim_rgb,
          lighting_fix, use_homography, resize_factor):
    
    if use_homography:
        from preprocessing.homography import homography
    if lighting_fix:
        from preprocessing.histogram_matching import light_diff_elimination

    #read the inputs
    image_1 = cv2.imread(input_path, 1)
    image_2 = cv2.imread(reference_path, 1)

    #we need the images to be the same size. resize_factor is for increasing or decreasing further the images
    new_shape = (int(resize_factor*0.5*(image_1.shape[1]+image_2.shape[1])), int(resize_factor*0.5*(image_1.shape[0]+image_2.shape[0])))
    image_1 = cv2.resize(image_1,new_shape, interpolation=cv2.INTER_AREA)
    image_2 = cv2.resize(image_2, new_shape, interpolation=cv2.INTER_AREA)
    global_variables.set_size(new_shape[0],new_shape[1])
    

    if use_homography:
        image2_registered, mask_registered, blank_pixels = homography(image_1, image_2)
    else:
        image2_registered  = image_2

    if use_homography:
        image_1[blank_pixels] = [0,0,0]
        image2_registered[blank_pixels] = [0, 0, 0]

    if (lighting_fix):
        image2_registered = light_diff_elimination(image_1, image2_registered)

    clustering_map, mse_array, size_array = compute_change_map(image_1, image2_registered, window_size=window_size,
                                                               clusters=n, pca_dim_gray= pca_dim_gray, pca_dim_rgb=pca_dim_rgb)

    clustering = [[] for _ in range(n)]
    for i in range(clustering_map.shape[0]):
        for j in range(clustering_map.shape[1]):
            clustering[int(clustering_map[i,j])].append([i,j])

    input_image = cv2.imread(input_path)
    input_image = cv2.resize(input_image,new_shape, interpolation=cv2.INTER_AREA)
    b_channel, g_channel, r_channel = cv2.split(input_image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    alpha_channel[:, :] = 50
    groups = find_group_of_accepted_classes_DBSCAN(mse_array)
    for group in groups:
        transparent_input_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        result = draw_combination_on_transparent_input_image(mse_array, clustering, group, transparent_input_image)
        
    return result


if __name__ == '__main__':

    
    input_path = "img/golden.jpg"
    reference_path = "img/diff1.jpg"
    n=15
    window_size = 5
    pca_dim_gray = 3
    pca_dim_rgb = 9
    
    lighting_fix = True
    use_homography = True
    resize_factor = float(1)
    
    
    result = main(input_path, reference_path, int(n), int(window_size),
         int(pca_dim_gray), int(pca_dim_rgb), bool(lighting_fix), bool(use_homography),
         float(resize_factor))
    
    cv2.imwrite('ACCEPTED_CLASSES'+'.png', result)
