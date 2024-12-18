import os
import cv2 
import pandas as pd
import numpy as np 
from tensorflow.keras.preprocessing.image import img_to_array, load_img


np.random.seed(624)
mList= [1.1, 2, 3.5]            #for which m values we want to compare masks


##################  Compare masks
class_names= ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def load_images_and_labels(class_dir, img_size):
    images = []
    image_names = []
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = load_img(image_path, target_size=img_size)
        image = img_to_array(image)
        images.append(image)
        image_names.append(image_name)
    return np.array(images), np.array(image_names)

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def compute_metrics(true_masks, test_masks):
    """
    Compute precision, recall, Jaccard index (IoU), and F1-score for two sets of masks.
    """
    metrics = {
        "precision": [],
        "recall": [],
        "jaccard": [],
        "f1": []
    }
    true_flat = true_masks.flatten()
    test_flat = test_masks.flatten()
    
    # Calculate metrics
    precision = precision_score(true_flat, test_flat, average='binary', zero_division=1)
    recall = recall_score(true_flat, test_flat, average='binary', zero_division=1)
    jaccard = jaccard_score(true_flat, test_flat, average='binary')
    f1 = f1_score(true_flat, test_flat, average='binary')
    
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["jaccard"].append(jaccard)
    metrics["f1"].append(f1)

    return metrics


for mval in mList: 
    dict_metrices_all_class= {
        'precision': [], 'recall': [], 'jacc':[], 'f1':[]
    }
    for cls in class_names: 
        # a) Load the groundtruth masks from pre-saved folder
        true_masks, mask_names= load_images_and_labels(f'../Data/segmentationMasks/{cls}/', (96,96))
        true_masks_grayscale = np.zeros((true_masks.shape[0], true_masks.shape[1], true_masks.shape[2]), dtype=np.uint8)
        for i in range(true_masks.shape[0]):
            true_masks_grayscale[i] = cv2.cvtColor(true_masks[i], cv2.COLOR_BGR2GRAY)
        
        _, true_masks_2D = cv2.threshold(true_masks_grayscale, 127, 255, cv2.THRESH_BINARY)

        # b) load my fuzzy masks
        mstr= 'm_' + str(mval)
        test_masks, mask_names= load_images_and_labels(f'../Data/saved_masks/{mstr}/{cls}/', (96,96))
        test_masks_2D= test_masks[..., 0]
        metrics= compute_metrics (true_masks_2D/255.0, test_masks_2D/255.0)

        print(f'Comparison metrics for m={mval}, class={cls}: {metrics}')

        dict_metrices_all_class['precision'].append(metrics['precision'])
        dict_metrices_all_class['recall'].append(metrics['recall'])
        dict_metrices_all_class['jacc'].append(metrics['jaccard'])
        dict_metrices_all_class['f1'].append(metrics['f1'])

    avg_prec= np.average(dict_metrices_all_class['precision'])
    avg_rec= np.average(dict_metrices_all_class['recall'])
    avg_jacc= np.average(dict_metrices_all_class['jacc'])
    avg_f1= np.average(dict_metrices_all_class['f1'])
    print(f'Across all class, average precision, recall, jaccard, f1 for m: {mval}= {avg_prec}, {avg_rec}, {avg_jacc}, {avg_f1}')
    


#############   In case masks aren't ordered in 7 folders, run this to order first
################### Order the segmentation masks in different folders and each of size 220x220 
def order_images( ):
    images_folder= '../Data/HAM10000_segmentations_lesion_tschandl'
    csv_file_path = '../Data/HAM10000_metadata.csv'
    output_folder = '../Data/segmentationMasks/' 

    data = pd.read_csv(csv_file_path)

    # Loop through each row in the CSV
    for index, row in data.iterrows():
        image_id = row['image_id']
        dx = row['dx']
        source_image_path = os.path.join(images_folder, f"{image_id}_segmentation.png")  

        target_folder = os.path.join(output_folder, str(dx))
        os.makedirs(target_folder, exist_ok=True)  # Create folder for dx if it doesn't exist
        target_image_path = os.path.join(target_folder, f"{image_id}_segmentation.png")

        if os.path.exists(source_image_path):
            img = cv2.imread(source_image_path)
        
            if img is not None:
                resized_img = cv2.resize(img, (220, 220))
                cv2.imwrite(target_image_path, resized_img)
            else:
                print(f"Failed to load image {image_id}.")
        else:
            print(f"Image {image_id} not found in source folder.")

    print("Image ordering done.")

#uncomment the line below to order the masks
#order_images()           

