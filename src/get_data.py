import os
import glob
import numpy as np
import cv2
from src.image import Image,ImageFactory 

def read_pts(filename):
    """
    Read landmarks .pts file safely.
    Returns None if the file is missing or empty.
    """
    
    if not os.path.exists(filename):
        return None

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        return None

    landmarks = []
    start_reading = False
    
    for line in lines:
        line = line.strip()
    
        if line == '{':
            start_reading = True
            continue
        if line == '}':
            break
            
        if start_reading:
            parts = line.split()
            if len(parts) >= 2:
                
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    landmarks.append([x, y])
                except ValueError:
                    continue
                
    if not landmarks:
        return None
        
    return np.array(landmarks)
def compute_mean_shape(landmarks_list):
    """
    Computes the mean shape from the training set of landmarks
    """
    normalized_shapes=[]
    
    for lm in landmarks_list:
        # For each landmark:
            # Compute the weight center and center all points
        center=np.mean(lm,axis=0)
        centered_lm=lm-center

        normalized_shapes.append(centered_lm)

    #Compute the men of all shapes
    mean_shape=np.mean(np.array(normalized_shapes), axis=0)
    
    return mean_shape

def get_data(data_folder, train_split=0.8):

    """
    Global prep function for our experiment:
        Load all pictures and landmarks, compute mean shape
        Converts picture into instances of the Image class
        Do the train test split

    return: (train_images_objects, test_images_objects)
    """
    
    # Load pictures
    img_extensions = ['*.jpg', '*.png', '*.jpeg']
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob.glob(os.path.join(data_folder, ext)))
        
    img_files.sort()

    all_data = [] 
    all_landmarks_for_mean = []


    for img_path in img_files:

        base_name = os.path.splitext(img_path)[0]
        pts_path = base_name + ".pts"
        
        pts_path = base_name + ".pts"
        landmarks = read_pts(pts_path)
        if landmarks is None:
            continue

        # Lecture Image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Lecture Landmarks
        landmarks = read_pts(pts_path)
        
        all_data.append({'img': img, 'lm': landmarks})
        all_landmarks_for_mean.append(landmarks)



    # Split Train / Test
    split_idx = int(len(all_data) * train_split)
    train_raw = all_data[:split_idx]
    test_raw = all_data[split_idx:]
    
    # Mean Shape on train set
    train_landmarks=[d['lm'] for d in train_raw]
    mean_shape=compute_mean_shape(train_landmarks)
    
    print(f"Data loaded: {len(train_raw)} Train, {len(test_raw)} Test.")

    factory = ImageFactory(mean_shape)

    train_objects = []
    for item in train_raw:
        obj=factory.create_image(item['img'], true_landmark=item['lm'], mode='train')
        train_objects.append(obj)
        
    test_objects = []
    for item in test_raw:
        # In test we init the landmark with Haar Cascade
        # But uses the true landmark to compute error
        obj = factory.create_image(item['img'], true_landmark=item['lm'], mode='test')
        test_objects.append(obj)
        
    return train_objects, test_objects, mean_shape