import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

class Image:
    """
    Class for images in order to store some info in a clean way:
        - the true landmark (during training mode)
        - the current landmark 
    """

    def __init__(self,image,current_landmark,true_landmark) -> None:
        self.image=image
        
        if len(self.image.shape)==3:
            self.image_gray=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.height,self.width,_=self.image.shape

        else:
            self.image_gray=self.image
            self.height,self.width=self.image.shape

        self.true_landmark=true_landmark
        self.current_landmark=current_landmark
    
    def set_landmark(self,landmark):
        self.current_landmark=landmark
        pass
    
    def feature_extraction(self,extraction_function):
        '''
        Extract features with function from the picture and current landmark
        It only analyse the keypoints at the landmark positions because that is what we want to regress
        Returns the descriptors from function 
        '''
        keypoints=[]
  
        for point in self.current_landmark:
            x, y = point
            # size is 32 like in the paper
            kp = cv2.KeyPoint(x=float(x), y=float(y), size=32)
            keypoints.append(kp)

        keypoints,descriptors=extraction_function.compute(self.image_gray,keypoints)
        if descriptors is None:
            return np.zeros(len(self.current_landmark)*128)
             
        return descriptors.flatten()
          
    def show(self, show_true=False, show_current=True):

        """Function to show image and landmark"""

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(self.image_gray, cv2.COLOR_BGR2RGB))
        
        if show_true and self.true_landmark is not None:
            plt.scatter(self.true_landmark[:, 0], self.true_landmark[:, 1], 
                        c='green', s=10, marker='.', label='Ground Truth')
            
        if show_current and self.current_landmark is not None:
            plt.scatter(self.current_landmark[:, 0], self.current_landmark[:, 1], 
                        c='red', s=10, marker='.', label='Prediction')


        plt.legend()
        plt.axis('off')
        plt.show()


class ImageFactory:
    """
    Factory class responsible for preparing data and instantiating Image objects.
    It handles:
    - Bounding Box calculation (from ground truth for dataset image or face detection for my images)
    - Mean Shape alignment (initialization)
    """

    def __init__(self,mean_shape):
        
        # Like in the paper we use a face detector to get a bounding box around faces to initiate mean landmark.
        # This improves and speeds up training
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"File not found : {xml_path}. Need the .xml file in src/")

        self.face_cascade = cv2.CascadeClassifier(xml_path)
        self.mean_shape=mean_shape


    def create_image(self, image, true_landmark=None, mode='train'):
        """
        Main method to create a clean Image object.
         image: Raw cv2 image
         true_landmark: Ground truth landmarks (mandatory for 'train', optional for 'test')
         'train' (uses true_landmark for init) or 'test' (uses face detector)
        """
        bbox=None
        if mode=='train':
            if true_landmark is None:
                raise ValueError("In 'train' mode, true_landmark is required.")
            bbox = self._compute_bbox_from_landmarks(true_landmark)
                    
        elif mode == 'test':
            faces = self._detect_face(image)
            
            if len(faces) > 0:
                if true_landmark is not None:
                    # Select the face closest to the ground truth
                    bbox = self._select_best_bbox(faces, true_landmark)
                else:
                    # Blind test: take the first one
                    bbox = faces[0]

            # Fallback: if no face, align at center

            if bbox is None:
                    h,w=image.shape[:2]
                    bbox= (w//4,h//4,w//2,h//2)

        # Put the mean shape inside the box
        initial_landmark = self._align_mean_shape(bbox)

        # Ouput the clean Image instance
        return Image(image, initial_landmark, true_landmark)

    def _compute_bbox_from_landmarks(self, landmarks):
        """Calculates bbox (x, y, w, h) from points."""
        min_x, min_y = np.min(landmarks, axis=0)
        max_x, max_y = np.max(landmarks, axis=0)
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _detect_face(self, image):
        """Uses OpenCV Haar Cascade to find the face bbox."""

        if len(image.shape)==3:
            gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray=image
            
        faces=self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces # Returns the list of all detected faces

    def _select_best_bbox(self, faces, true_landmark):
        """
        Selects the detected bbox that is spatially closest to the true landmark center.
        """
        true_center = np.mean(true_landmark, axis=0)
        best_face = None
        min_dist = float('inf')

        for (x, y, w, h) in faces:
            face_center = np.array([x + w/2, y + h/2])
            dist = np.linalg.norm(true_center - face_center)
            
            if dist < min_dist:
                min_dist = dist
                best_face = (x, y, w, h)
        
        return best_face

    def _align_mean_shape(self, bbox):
        """
        Aligns and scales the centered mean_shape to the center of the provided bbox.
        """
        x,y,w,h=bbox
        
        bbox_center_x=x+w / 2
        bbox_center_y=y+h/ 2
        mean_min=np.min(self.mean_shape,axis=0)
        mean_max=np.max(self.mean_shape,axis=0)
        mean_center_x=(mean_min[0]+mean_max[0])/2
        mean_center_y=(mean_min[1]+mean_max[1])/2
        mean_width = mean_max[0] - mean_min[0]
        mean_height = mean_max[1] - mean_min[1]
        scale_x = w / mean_width if mean_width > 0 else 1.0
        scale_y = h / mean_height if mean_height > 0 else 1.0
        scale = (scale_x + scale_y) / 2
        scaling_factor = scale * 0.9 
        aligned_shape = self.mean_shape.copy().astype(np.float32)
        aligned_shape[:, 0] -= mean_center_x
        aligned_shape[:, 1] -= mean_center_y
        aligned_shape *= scaling_factor
        aligned_shape[:, 0] += bbox_center_x
        aligned_shape[:, 1] += bbox_center_y

        pixel_offset=h*0.15 # Important because face detector is around all face but the landmark are on the bottom of the face !
        aligned_shape[:, 1] += pixel_offset
        
        return aligned_shape