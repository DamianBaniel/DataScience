#  Cancer classification
 

## Project Overview
   The objective of this project was to classify breast tumors as benign (B) or malignant (M) using the Wisconsin Breast Cancer Dataset. The project included:
     Exploratory Data Analysis (EDA)
     Model training and evaluation
     Performance comparison of multiple algorithms

# Data Exploration
  Initial analysis focused on understanding the distribution of features.

 <img width="1160" alt="Figure 2025-06-25 115414 (0)" src="https://github.com/user-attachments/assets/7a9ce836-379c-427c-81fd-b56bc7c873d8" />

   - The features are right-skewed and vary significantly in scale.
   - Some features like V4 and V29 have wide ranges, while others are tightly clustered.
   - Standardization or normalization is recommended for distance-based models.
    
# Feature Distributions

  Models Used
  I trained and evaluated the following machine learning models:
  
   Logistic Regression
   K-Nearest Neighbors (KNN)
   Gaussian Naive Bayes
   Random Forest
   Gradient Boosting
   Support Vector Classifier (SVC)

# Performance Evaluation
   Confusion Matrices

<img width="313" alt="Figure 2025-06-25 115414 (1)" src="https://github.com/user-attachments/assets/0201d149-e065-4a87-9dad-ef3213f69c6c" /><img width="313" alt="Figure 2025-06-25 115414 (4)" src="https://github.com/user-attachments/assets/35b65446-cd32-4950-9d30-17e87c6bc330" />
<img width="313" alt="Figure 2025-06-25 115414 (6)" src="https://github.com/user-attachments/assets/c2c46736-7941-4145-ac18-2fd7664a2499" /><img width="313" alt="Figure 2025-06-25 115414 (5)" src="https://github.com/user-attachments/assets/cd6f937a-2df8-494f-a27a-37647fd9ebbe" />
<img width="313" alt="Figure 2025-06-25 115414 (3)" src="https://github.com/user-attachments/assets/afa7d7f9-0e08-43b2-a503-f1942512d985" /><img width="313" alt="Figure 2025-06-25 115414 (2)" src="https://github.com/user-attachments/assets/3761f2cd-7aae-437a-8fbb-b6b5a7ae108c" />


# ROC Curve Comparison

 <img width="484" alt="Figure 2025-06-25 115414 (7)" src="https://github.com/user-attachments/assets/145fa87f-dbd6-4403-bf22-40f871ce1c96" />

# Robustness Checks
    Gaussian noise (±10% σ): all models retained ≥ 90 % accuracy on noisy test data.
    Outlier injection (5% extreme spikes): tree-based methods (RF, GB) showed the smallest drop in accuracy, highlighting their resilience to outliers.

# Overall Conclusions
    Best overall performer:
     Random Forest achieved the highest ROC AUC (0.99) and malignant recall (92 %).
    Best recall-on-malignancy:
     Logistic Regression and SVC minimized false negatives (4), critical for cancer screening.    
    Fast & interpretable contender: 
     GaussianNB offers near-top accuracy with minimal computational cost and clear per-feature distributions. 
    Clinical priority takeaway:    
     Prioritize models that minimize malignant false negatives, even at the cost of a few extra benign false positives.
    


