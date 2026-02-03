from sklearn.linear_model import LinearRegression,Ridge
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt


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
    
    def fit(self,image_list,test_data):
        """
        Whole function for fitting the SDM
        """

        print(f"Training SDM on {len(image_list)} images for {self.n_step} steps.")

        self.coef_list=[]
        self.intercept_list=[]
        self.nme_history=[]

        for step in range(self.n_step):
            
            # Fit one step
            R_k,b_k=self.step_fit(image_list)
            self.coef_list.append(R_k)
            self.intercept_list.append(b_k)

            # Update training set for next step
            self.update_dataset(image_list,R_k,b_k)
            current_nme, _ = self.evaluate(test_data)
            self.nme_history.append(current_nme)

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
        Computes the NME (Normalized Mean Error) on test set
        """
        all_errors = []
        
        for img_obj in image_list:
            pred_img = self.predict(img_obj)
            
            pred_lm = pred_img.current_landmark
            true_lm = img_obj.true_landmark
            
            # Compute euclidian distance then normalize by the eye distance
            diff = pred_lm - true_lm
            dist = np.linalg.norm(diff, axis=1) 
            mean_error_pixels = np.mean(dist)

            
            if len(true_lm) == 68:
                left_eye_center = np.mean(true_lm[36:42], axis=0)
                right_eye_center = np.mean(true_lm[42:48], axis=0)
                normalization_factor = np.linalg.norm(left_eye_center - right_eye_center)
            else:
                # Fallback: use the diagonal of the bounding box
                w = np.max(true_lm[:,0]) - np.min(true_lm[:,0])
                h = np.max(true_lm[:,1]) - np.min(true_lm[:,1])
                normalization_factor = np.sqrt(w**2 + h**2)

            if normalization_factor == 0: normalization_factor = 1.0
                
            #(NME)
            nme = mean_error_pixels / normalization_factor
            all_errors.append(nme)
            
        final_score = np.mean(all_errors)
        print(f"Mean NME  : {final_score:.4f} ({final_score*100:.2f}%)")
        
        return final_score, all_errors

    def evaluate_in_pixels(self, test_data):
        """
        Compute the euclidian distance 
        """
        total_pixel_error = 0
        total_points = 0
        max_error = 0
        worst_image_idx = -1
        
        errors_per_image = []

        for idx, img_obj in enumerate(test_data):
            # Prediction
            pred_img = self.predict(img_obj)
            pred_lm = pred_img.current_landmark
            true_lm = img_obj.true_landmark
            
            # Euclidian distance
            diff_vectors=pred_lm-true_lm
            distances=np.linalg.norm(diff_vectors, axis=1)
            mean_img_error=np.mean(distances)
            max_img_error=np.max(distances)
            
            errors_per_image.append(mean_img_error)
            total_pixel_error += np.sum(distances)
            total_points += len(distances)
            
            # Worst case tracking
            if mean_img_error > max_error:
                max_error = mean_img_error
                worst_image_idx = idx

        global_mean_pixel_error = total_pixel_error / total_points
    
        
        return global_mean_pixel_error, errors_per_image, worst_image_idx, max_error
    
    def visualize_error_vectors(self, test_data, image_indices=None):
        """
        Print error vectors, from prediction to truth
        """
        if image_indices is None:
            image_indices = np.random.choice(len(test_data),2, replace=False)
            
        for idx in image_indices:
            img_obj = test_data[idx]
            pred_img = self.predict(img_obj)
            
            pred_lm = pred_img.current_landmark.astype(int)
            true_lm = img_obj.true_landmark.astype(int)

            if len(img_obj.image.shape) == 3:
                canvas = cv2.cvtColor(img_obj.image.copy(), cv2.COLOR_BGR2RGB)
            else:
                canvas = cv2.cvtColor(img_obj.image.copy(), cv2.COLOR_GRAY2RGB)
        
            diff = np.linalg.norm(pred_lm - true_lm, axis=1)
            err_px = np.mean(diff)


            for i in range(len(pred_lm)):
                pt_pred = tuple(pred_lm[i])
                pt_true = tuple(true_lm[i])

                cv2.line(canvas, pt_pred, pt_true, (255, 0, 255), 1) 

                cv2.circle(canvas, pt_pred, 2, (0, 255, 255), -1)
                cv2.circle(canvas, pt_true, 2, (0, 255, 0), -1)   

            plt.figure(figsize=(8, 8))
            plt.imshow(canvas)
            plt.title(f"Image {idx} - Mean error: {err_px:.1f} px\n Pink lines = Deplacement error")
            plt.axis('off')
            plt.legend(['Error', 'Pred (Cyan)', 'Truth (Vert)']) 
            plt.show()

    def graph_nme_hist(self):
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1,len(self.nme_history)+1), self.nme_history, marker='o', linestyle='-', color='b', label='test error')
        plt.title('Evolution of NME during training')
        plt.xlabel('Step')
        plt.ylabel('NME')
        plt.grid(True)
        plt.legend()
        plt.show()

        return self.nme_history