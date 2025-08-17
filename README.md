🧠 Support Vector Machines (SVM) for Breast Cancer Classification
📌 Objective

Use Support Vector Machines (SVMs) for binary classification of breast cancer (Malignant vs. Benign).
The project explores both linear and non-linear (RBF kernel) SVMs, along with hyperparameter tuning and cross-validation.

⚙️ Tools & Libraries

Python 🐍

Scikit-learn (SVM, preprocessing, model selection, metrics)

NumPy

Matplotlib

Pandas

📂 Dataset

We use the Breast Cancer dataset (breast-cancer.csv), which contains features extracted from cell nuclei of breast cancer biopsies.

Target Column: diagnosis

M → Malignant (cancerous)

B → Benign (non-cancerous)

Features: Numerical measurements of cell nuclei (mean radius, texture, perimeter, smoothness, etc.).

🚀 Steps in the Project

Load & Explore Data – Import dataset and check shape, features.

Preprocessing – Encode labels (M=1, B=0), scale features using StandardScaler.

Train Models –

Linear SVM (kernel='linear')

RBF Kernel SVM (kernel='rbf')

Evaluate Models – Accuracy, confusion matrix, classification report.

Hyperparameter Tuning – Use GridSearchCV for tuning C and gamma.

Cross-validation – Perform k-fold CV for robust evaluation.

Visualization – Plot decision boundary (with 2 selected features).

📊 Results

Linear SVM gives good separation but may underperform for complex boundaries.

RBF Kernel SVM usually provides better accuracy by handling non-linear relationships.

Hyperparameter tuning (C, gamma) significantly improves results.

Cross-validation ensures the model generalizes well.

🖥️ How to Run

Clone this repository or download the files.

Install dependencies:

pip install scikit-learn pandas numpy matplotlib


Run the Python script:

python svm_classification.py

📈 Example Visualization

The project also includes a decision boundary plot (using 2 features for 2D visualization):

Blue vs Red regions represent classification boundaries.

Circles indicate data points (malignant/benign).

📌 Key Learnings

Linear SVM works well when data is linearly separable.

RBF Kernel SVM handles non-linear decision boundaries better.

C (regularization) and gamma (RBF spread) are critical hyperparameters.

Cross-validation prevents overfitting and gives a better estimate of model performance.

✅ Next Steps

Try Polynomial kernel SVM.

Perform feature selection to reduce dimensionality.

Extend project to a real-time cancer prediction web app (Flask/Streamlit).

📌 Author: Bhumika Bhat
