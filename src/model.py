from sklearn.linear_model import LinearRegression
from src.image import Image
import numpy as np
import cv2 as cv


class SDM:

    def __init__(self,n_step=4) -> None:
        self.n_step=n_step
        self.coef_list=[]
        self.intercept_list=[]
        
        pass

    def step_fit(self,pic_set):
        """
        Do a step of fitting, like in the paper the model:
            Compute the step needed to go from current landmark to true landmark
            Try to estimate this step with a Lin Reg on the extracted feature at the current landmark

        """
        X_step=[]
        y_step=[]

        for pic in pic_set:

            # First compute the target step that need to be estimated: diff between true and current landmark
            current_landmark=pic.current_landmark
            true_landmark=pic.true_landmark
            target_landmark=true_landmark-current_landmark

            # Compute feature at current position
            extracted_landmark=pic.feature_extraction()
            X_step.append(extracted_landmark)
            y_step.append(target_landmark.flatten())


        X_step=np.array(X_step)
        y_step=np.array(y_step)

        model=LinearRegression()
        model.fit(X_step,y_step)

        return model.coef_,model.intercept_
    
    def update_dataset(self,pic_set):
        """
        Update the dataset by applying the current step to all pictures
        """
        for pic in pic_set:
        
            R=self.coef_list[-1]
            b=self.intercept_list[-1]
            phi=pic.feature_extraction()
            delta_x=R@phi+b
            
            pic.current_landmark+= delta_x.reshape(-1, 2)
        pass
    
    def fit(self,pic_set):
        """
        Whole function for fitting the SDM
        """
        self.coef_list=[]
        self.intercept_list=[]
        data_set=pic_set.copy()

        for step in range(self.n_step):

            # Fit one step
            R_k,b_k=self.step_fit(data_set)
            self.coef_list.append(R_k)
            self.intercept_list.append(b_k)

            # Update training set for next step
            self.update_dataset(data_set)
    
        return self.coef_list,self.intercept_list

