# Multivariate Cardiac Risk Analysis

## Project Overview
This project performs a comprehensive **multivariate numerical data analysis** on the world-renowned **Cleveland Heart Disease Database**. The primary goal is to predict the presence of heart disease in patients using Machine Learning and to extract diagnostic insights from clinical data.

## 🛠️ Technologies Used

* **Python**: The core programming language used for analysis and modeling.
* **Pandas & NumPy**: Used for data manipulation, handling the CSV dataset, and numerical operations.
* **Matplotlib & Seaborn**: Used for generating the Exploratory Data Analysis (EDA) charts (correlation heatmaps, bar plots).
* **Scikit-Learn**: The primary library for implementing machine learning algorithms (Random Forest, SVM, KNN, Decision Trees).
* **TensorFlow / Keras**: Used to build and train the Neural Network (Deep Learning) model.
* **Jupyter Notebook**: The interactive environment used to run the code and visualize results.

The project addresses two major tasks:
1.  **Predictive Task:** Build a classification model to determine if a patient has heart disease based on 14 clinical attributes.
2.  **Diagnostic Task:** Extract experimental insights to better understand the relationship between various physiological factors and cardiac health.



---

## Dataset Information
This is a **multivariate dataset**, involving a variety of separate mathematical and statistical variables. While the original database contains 76 attributes, this project utilizes the standard subset of **14 attributes** recognized globally by ML researchers for its high predictive relevance.

### The 14 Key Attributes:
* **Age**: Patient's age.
* **Sex**: (1 = male; 0 = female).
* **Chest Pain Type (CP)**: Typical angina, Atypical angina, Non-anginal pain, Asymptomatic.
* **Trestbps**: Resting blood pressure (in mm Hg).
* **Chol**: Serum cholesterol in mg/dl.
* **Fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
* **Restecg**: Resting electrocardiographic results.
* **Thalach**: Maximum heart rate achieved.
* **Exang**: Exercise-induced angina (1 = yes; 0 = no).
* **Oldpeak**: ST depression induced by exercise relative to rest.
* **Slope**: The slope of the peak exercise ST segment.
* **CA**: Number of major vessels (0-3) colored by fluoroscopy.
* **Thal**: Thalassemia status (3 = normal; 6 = fixed defect; 7 = reversible defect).
* **Target**: Heart disease status (1 = Presence, 0 = Absence).

---

## Project Steps & Experimental Findings

### 1. Exploratory Data Analysis (EDA)
We began by analyzing the distribution of the target variable and understanding how features like gender and chest pain correlate with heart disease.

#### Target Class Balance
![Target Distribution](./eda_target_distribution.png)
*Figure 1: Distribution of the target variable. The dataset is relatively balanced between patients with and without heart disease.*

#### Gender & Heart Disease
![Sex vs Target](./eda_sex_vs_target.png)
*Figure 2: Heart disease rates by gender. The data suggests a variation in risk between male (1) and female (0) patients.*

#### Impact of Chest Pain Type
![Chest Pain Analysis](./eda_cp_vs_target.png)
*Figure 3: Correlation between Chest Pain Type (CP) and Heart Disease. Asymptomatic and non-anginal pain types show higher correlation with positive diagnoses.*

### 2. Data Preprocessing
* **Feature-Target Split**: Separating the "target" column from the 13 clinical predictors.
* **Train-Test Split**: Dividing the data into **80% training** and **20% testing** sets to ensure the models are evaluated on unseen data.

### 3. Model Implementation
We benchmarked seven different machine learning algorithms to identify the most accurate diagnostic tool:
* **Logistic Regression**: Baseline binary classification.
* **Naive Bayes**: Probabilistic prediction.
* **Support Vector Machine (SVM)**: Optimal hyperplane separation.
* **K-Nearest Neighbors (KNN)**: Proximity-based classification.
* **Decision Tree**: Tree-based logic with optimized `random_state`.
* **Random Forest**: Ensemble of trees for robust accuracy.
* **Neural Network**: Deep learning multi-layer perceptron.

---

## Results and Benchmarking

The models were evaluated based on accuracy. The **Random Forest** and **Decision Tree** algorithms demonstrated the highest predictive power for this specific dataset.

### Final Accuracy Scores:

| Algorithm | Accuracy Score |
| :--- | :--- |
| **Random Forest** | **100.0%** |
| Decision Tree | 98.36% |
| Support Vector Machine | 88.52% |
| Logistic Regression | 86.89% |
| Naive Bayes | 82.79% |
| Neural Network | 81.15% |
| K-Nearest Neighbors | 72.13% |

### Model Performance Visualization
![Model Comparison](./model_accuracy_comparison.png)
*Figure 4: Final accuracy comparison of all 7 Machine Learning models. Ensemble methods (Random Forest, Decision Tree) demonstrated superior predictive power.*

---

## Conclusion
This project demonstrates that a subset of 14 clinical variables from the Cleveland Database is highly sufficient for building a robust predictive tool. By analyzing these multivariate interactions, we can provide significant diagnostic insights that could assist in early medical intervention.

## How to Run
1.  **Clone the Repo**: `git clone [your-repo-link]`
2.  **Install Dependencies**: `pip install numpy pandas matplotlib seaborn scikit-learn tensorflow`
3.  **Run**: Execute the `Heart_disease_prediction.ipynb` notebook to reproduce these results.