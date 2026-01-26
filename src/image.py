import cv2 
import numpy as np
import matplotlib.pyplot as plt



class Image:
    """
    Class for images in order to store some info in a clean way:
        the true landmark 
        the current landmark
    """
    def __init__(self,image,true_landmark) -> None:
        self.image=image
        self.image_gray=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        if len(self.image.shape)==3:
            self.height,self.width,_=self.image.shape
        else:
            self.height,self.width=self.image.shape

        self.true_landmark=true_landmark
        self.landmarks_hist={'true':true_landmark}
        self.extraction_function=cv2.SIFT_create()
        self.current_landmark=None
        pass


    def set_current_landmark(self,landmark):
        n=len(self.landmarks_hist.keys())
        self.current_landmark=landmark
        self.landmarks_hist[n]=landmark
        pass

    def _compute_bbox_from_landmarks(self, landmarks):
        """Calcule le rectangle englobant (x, y, w, h) autour des points."""
        min_x, min_y = np.min(landmarks, axis=0)
        max_x, max_y = np.max(landmarks, axis=0)
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def feature_extraction(self):
        '''
        Extract features with function from the picture and current landmark
        It only analyse the keypoints at the landmark positions becauser that is what we want to regress
        Returns the descriptors from function 
        '''
        keypoints = []
        for point in self.current_landmark:
            x,y=point
            kp = cv2.KeyPoint(x=float(x), y=float(y), size=32) # size 32 is arbitrary here but with face landamrk
            keypoints.append(kp)

        keypoints,descriptors= self.extraction_function.compute(self.image_gray,keypoints)
        
        return descriptors.flatten()
        
    def show(self, show_true=False, show_current=True):
        """Function to show image and landmark"""

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        
        if show_true and self.true_landmark is not None:
            plt.scatter(self.true_landmark[:, 0], self.true_landmark[:, 1], 
                        c='green', s=10, marker='.', label='Ground Truth')
            
        if show_current and self.current_landmark is not None:
            plt.scatter(self.current_landmark[:, 0], self.current_landmark[:, 1], 
                        c='red', s=10, marker='.', label='Prediction')

        # Dessiner la bbox
        if self.bbox is not None:
            x, y, w, h = self.bbox
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
            plt.gca().add_patch(rect)

        plt.legend()
        plt.axis('off')
        plt.show()

