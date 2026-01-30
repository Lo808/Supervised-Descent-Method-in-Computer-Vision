from sklearn.linear_model import LinearRegression
import numpy as np
import cv2


class SDM:
    def __init__(self,n_step=4,extraction_function=cv2.SIFT_create()) -> None:
        """
        n_step is the number of step of fitting, for each step we estimated the descent matrix
            the paper advises n_step around 4-5
        """
        self.n_step=n_step
        self.coef_list=[]
        self.intercept_list=[]
        self.extraction_function=extraction_function
    
        pass

    def step_fit(self,image_list):
        """
        Do a step of fitting: 
            The model computes the step needed to go from current landmark to true landmark
            Try to estimate this step with a Linear Regression on the extracted feature at the current landmark

        """
        X_step=[]
        y_step=[]

        for pic in image_list:
            # First compute the target step that need to be estimated: diff between true and current landmark
            current_landmark=pic.current_landmark
            true_landmark=pic.true_landmark
            target_landmark=true_landmark-current_landmark

            # Compute feature at current position
            extracted_landmark=pic.feature_extraction(self.extraction_function)
            X_step.append(extracted_landmark)
            y_step.append(target_landmark.flatten())


        X_step=np.array(X_step)
        y_step=np.array(y_step)

        model=LinearRegression()
        model.fit(X_step,y_step)

        return model.coef_,model.intercept_
    
    def update_dataset(self,image_list,R,b):
        """
        Update the picture by applying the estimated step
        The new picture is then used to calculate the next step
        """
        
        for pic in image_list:
            phi=pic.feature_extraction(self.extraction_function)
            delta_x=R@phi+b
            pic.current_landmark+= delta_x.reshape(-1, 2)


    
    def fit(self,image_list):
        """
        Whole function for fitting the SDM
        """

        print(f"Training SDM on {len(image_list)} images for {self.n_step} steps.")

        self.coef_list=[]
        self.intercept_list=[]

        for _ in range(self.n_step):
            
            # Fit one step
            R_k,b_k=self.step_fit(image_list)
            self.coef_list.append(R_k)
            self.intercept_list.append(b_k)

            # Update training set for next step
            self.update_dataset(image_list,R_k,b_k)
    
        return self.coef_list,self.intercept_list
    

    def predict(self,single_image):
        """
        Method to predict one image using the found matrixes
        """

        for R_k,b_k in zip(self.coef_list,self.intercept_list):
            phi=single_image.feature_extraction(self.extraction_function)
            delta_x=R_k@phi+b_k
            single_image.current_landmark+= delta_x.reshape(-1, 2)

        return single_image.current_landmark

