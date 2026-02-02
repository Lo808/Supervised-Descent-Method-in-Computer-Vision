from sklearn.linear_model import LinearRegression,Ridge
import numpy as np
import cv2
import copy

class SDM:
    def __init__(self,n_step=4,extraction_function=cv2.SIFT_create(),model=LinearRegression()) -> None:
        """
        n_step is the number of step of fitting, for each step we estimated the descent matrix
            the paper advises n_step around 4-5
        """
        self.n_step=n_step
        self.coef_list=[]
        self.intercept_list=[]
        self.extraction_function=extraction_function
        self.model=model
    
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

        model=self.model
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

        for step in range(self.n_step):
            
            # Fit one step
            R_k,b_k=self.step_fit(image_list)
            self.coef_list.append(R_k)
            self.intercept_list.append(b_k)

            # Update training set for next step
            self.update_dataset(image_list,R_k,b_k)

            print(f"Step {step+1} done.")
    
        return self.coef_list,self.intercept_list
    
    def predict(self,single_image):
        """
        Method to predict one image using the found matrixes
        """
        new_image=copy.deepcopy(single_image)

        for R_k,b_k in zip(self.coef_list,self.intercept_list):
            phi=new_image.feature_extraction(self.extraction_function)
            delta_x=R_k@phi+b_k
            new_image.current_landmark+= delta_x.reshape(-1, 2)

        return new_image


    def evaluate(self, image_list):
        """
        Calcule le NME (Normalized Mean Error) sur une liste d'images de test.
        Retourne :
            - mean_nme : L'erreur moyenne sur tout le dataset (le score final).
            - all_errors : La liste des erreurs par image (utile pour tracer la courbe CED).
        """
        all_errors = []
        
        print(f"Évaluation sur {len(image_list)} images...")
        
        for img_obj in image_list:
            # 1. Prédire les landmarks
            # (predict fait déjà une deepcopy, donc c'est safe)
            pred_img = self.predict(img_obj)
            
            pred_lm = pred_img.current_landmark
            true_lm = img_obj.true_landmark
            
            # 2. Calculer l'erreur brute (Euclidean Distance)
            # diff = sqrt((x1-x2)^2 + (y1-y2)^2) pour chaque point
            diff = pred_lm - true_lm
            dist = np.linalg.norm(diff, axis=1) # Tableau de 68 distances
            mean_error_pixels = np.mean(dist)

            
            if len(true_lm) == 68:
                left_eye_center = np.mean(true_lm[36:42], axis=0)
                right_eye_center = np.mean(true_lm[42:48], axis=0)
                normalization_factor = np.linalg.norm(left_eye_center - right_eye_center)
            else:
                # Fallback générique si ce n'est pas 68 points :
                # On utilise la diagonale de la bounding box des vrais landmarks
                w = np.max(true_lm[:,0]) - np.min(true_lm[:,0])
                h = np.max(true_lm[:,1]) - np.min(true_lm[:,1])
                normalization_factor = np.sqrt(w**2 + h**2)
            
            # Sécurité division par zéro
            if normalization_factor == 0: normalization_factor = 1.0
                
            #(NME)
            nme = mean_error_pixels / normalization_factor
            all_errors.append(nme)
            
        final_score = np.mean(all_errors)
        print(f"Mean NME  : {final_score:.4f} ({final_score*100:.2f}%)")
        
        return final_score, all_errors
