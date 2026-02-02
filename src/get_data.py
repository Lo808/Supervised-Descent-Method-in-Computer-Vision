import os
import glob
import numpy as np
import cv2
from src.image import Image,ImageFactory 
import random

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

def perturb_bbox(bbox, n_perturbations=10):
    """
    Function to generate perturbed images.
    In order to learn more robustly, the authors suggest sampled perturbed image with shifted mean face
    """
    x,y,w,h=bbox
    perturbed_bboxes=[]

    # Start with original box
    perturbed_bboxes.append(bbox)
    
    for _ in range(n_perturbations - 1):
        # Shift with a random distribution
        trans_x=np.random.uniform(-0.05,0.05)*w
        trans_y= np.random.uniform(-0.05,0.05)*h
        
        #Also scale randomly
        scale = np.random.uniform(0.95, 1.05)
        
        # New box calculation
        new_w=w*scale
        new_h =h*scale
        new_x=x+trans_x-(new_w-w)/2
        new_y=y+trans_y-(new_h-h)/2
        
        perturbed_bboxes.append((new_x,new_y,new_w,new_h))
        
    return perturbed_bboxes

def calculate_iou(bbox1, bbox2):
    """
    Computes the Intersection over Union (IoU) between 2 bounding boxes in order to remove outliers
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1+w1, y1+h1
    box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2+w2, y2+h2
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

def filter_usable_images(raw_data, factory, iou_threshold=0.3):
    """
    Filtre le dataset pour ne garder que les images où le détecteur de visage
    trouve une bounding box cohérente avec la vérité terrain.
    """
    cleaned_data = []
    rejected_count = 0
    
    for item in raw_data:
        true_lm = item['lm']
        img = item['img']
        
        # Truth
        gt_bbox = factory._compute_bbox_from_landmarks(true_lm)
        
        #Detected ifaces
        faces = factory._detect_face(img)
        keep = False

        if len(faces) > 0:
            # Select best box and check coherence
            best_face = factory._select_best_bbox(faces, true_lm)
            iou = calculate_iou(gt_bbox, best_face)
            if iou > iou_threshold:
                keep = True
        if keep:
            cleaned_data.append(item)
        else:
            rejected_count += 1
            
    return cleaned_data, rejected_count

def get_data(data_folder, train_split=0.8,n_perturbations=10):
    """
    Global prep function for our experiment:
        Load all pictures and landmarks, compute mean shape, remove outliers
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

    temp_landmarks = [d['lm'] for d in all_data]
    temp_mean=compute_mean_shape(temp_landmarks)
    temp_factory=ImageFactory(temp_mean)
    
    clean_data, n_rejected = filter_usable_images(all_data, temp_factory, iou_threshold=0.3)
    random.shuffle(clean_data)
    
    print(f"Data Cleaning: Kept {len(clean_data)} images. Rejected {n_rejected} bad detections.")

    # Split Train / Test
    split_idx = int(len(clean_data) * train_split)
    train_raw = clean_data[:split_idx]
    test_raw = clean_data[split_idx:]
    
    # Mean Shape on train set
    train_landmarks=[d['lm'] for d in train_raw]
    mean_shape=compute_mean_shape(train_landmarks)
    
    print(f"Data loaded: {len(train_raw)} Train, {len(test_raw)} Test.")

    factory = ImageFactory(mean_shape)

    train_objects = []
    print(f"Generating augmented training data ({n_perturbations} per image)")
    
    for item in train_raw:
        true_lm=item['lm']
        img=item['img']
        
        # Bbox Ground Truth 
        gt_bbox = factory._compute_bbox_from_landmarks(true_lm)

        # Use the helper function to get valid perturbations
        bboxes = perturb_bbox(gt_bbox, n_perturbations)
        
        for bbox in bboxes:
            # We verify that the generated box is not completely off
            if calculate_iou(gt_bbox, bbox) > 0.5:
                initial_landmark = factory._align_mean_shape(bbox)
                obj = Image(img, initial_landmark, true_lm)
                train_objects.append(obj)

            
    test_objects = []
    for item in test_raw:
        obj = factory.create_image(item['img'], true_landmark=item['lm'], mode='test')
        test_objects.append(obj)
            
    return train_objects, test_objects, mean_shape


