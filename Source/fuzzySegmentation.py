import os
import numpy as np
import cv2
from skfuzzy.cluster import cmeans
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt 


# Parameters
tval= 'test'       #want to segment train data or test data?
m= 1.1

# Modify the directories according to your needs
img_size = (96, 96)
input_folder = '../Data/' + tval  
output_folder = '../Data/segmented_images_' + tval  
output_folder_mask= '../Data/saved_masks'
os.makedirs(output_folder, exist_ok=True)


def load_images_and_labels(data_dir, img_size):
    images = []
    labels = []
    image_names = []
    class_names = os.listdir(data_dir)
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = load_img(image_path, target_size=img_size)
            image = img_to_array(image)
            images.append(image)
            labels.append(class_name)
            image_names.append(image_name)
    return np.array(images), np.array(labels), np.array(image_names)



def extract_lesion(image, segmented_image, target_cluster):
    """
    Extract the lesion portion from the original image using the cluster mask.
    :param image: Original RGB image
    :param segmented_image: 2D array of cluster labels
    :param target_cluster: Cluster label corresponding to the lesion
    :return: RGB image with only the lesion portion
    """
    extracted = np.zeros_like(image)   
    mask = (segmented_image == target_cluster)   
    extracted[mask] = image[mask]  
    return extracted, mask 


def fuzzy_cmeans_segmentation_rgb(image, m, n_clusters=2):
    """
    Perform fuzzy c-means clustering directly on the RGB values of the image.
    :param image: RGB image 
    :param m: Fuzziness parameter
    :param n_clusters: Number of clusters
    :return: Cluster labels reshaped into the image dimensions
    """
    h, w, c = image.shape
    img_flat = image.reshape(-1, c)
    img_flat_norm = img_flat.T  # Transpose to make it (3, n_pixels)

    cntr, u, _, _, _, _, _ = cmeans(img_flat_norm, n_clusters, m, error=0.005, maxiter=1000)

    cluster_labels = np.argmax(u, axis=0)
    segmented_image = cluster_labels.reshape(h, w)
    cluster_sizes = [(segmented_image == i).sum() for i in range(n_clusters)]
    tc= np.argmin(cluster_sizes)
    lesion, mask  = extract_lesion(image, segmented_image, target_cluster= tc)
    return lesion, mask 


images, labels, img_names = load_images_and_labels(input_folder, img_size)
for img, label, name in zip(images, labels, img_names):
    segmented_image, mask= fuzzy_cmeans_segmentation_rgb (img, m)
    
    output_subfolder = os.path.join(output_folder, label)
    os.makedirs(output_subfolder, exist_ok=True)
    output_path = os.path.join(output_subfolder, name)
    bgr_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR) 
    cv2.imwrite(output_path, bgr_image)
    print(f"Image written to {output_path}")

    #uncomment below to save mask
    # mval= 'm_' + str(m) 
    # output_subfolder = os.path.join(output_folder_mask, mval, label)
    # os.makedirs(output_subfolder, exist_ok=True)
    # maskName= name.split('.')[0] + '_segmentation.png'
    # output_path = os.path.join(output_subfolder, maskName)
    # mask = mask * 255.0
    # cv2.imwrite(output_path, mask)
    