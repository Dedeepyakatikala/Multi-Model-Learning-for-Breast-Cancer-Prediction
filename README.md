# Multi-Model-Learning-for-Breast-Cancer-Prediction
Breast cancer is one of the most prevalent causes of cancer-related deaths in women worldwide. Early and accurate diagnosis is essential for effective treatment. This project proposes a **dual-modality learning approach** that integrates:

- **Deep Learning (CNNs)** for image-based classification using mammograms and histopathology images.
- **Classical Machine Learning** for structured data analysis using algorithms like Logistic Regression, SVM, Random Forest, XGBoost, and Gradient Boosting on the WDBC dataset.

Our results show that while CNNs are effective for image analysis, simpler models like Logistic Regression yield superior performance on well-prepared structured datasets. This hybrid approach allows us to leverage the strengths of both data types and methodologies.

---
## 🧩 Project Workflow

### 🔬 Image-Based Prediction
1. **Datasets**:
   - **CBIS-DDSM** (mammogram images)
   - **Histopathological IDC Dataset**

2. **Pipeline**:
   - Image preprocessing (resizing, normalization, augmentation)
   - CNN model development and evaluation
   - Visualization: accuracy plots, confusion matrix

### 📊 Structured Data Prediction
1. **Dataset**:
   - **WDBC (Wisconsin Diagnostic Breast Cancer)**

2. **Pipeline**:
   - Data cleaning, encoding, feature scaling
   - ML model training: Logistic Regression, SVM, KNN, etc.
   - Evaluation using accuracy, F1-score, ROC curves

---

## 🔧 Tech Stack

| Layer        | Technology                     |
|--------------|--------------------------------|
| **Language** | Python                         |
| **DL**       | TensorFlow, Keras              |
| **ML**       | scikit-learn, XGBoost          |
| **Visualization** | matplotlib, seaborn       |
| **Datasets** | CBIS-DDSM, IDC Histology, WDBC|

---

## 📂 Datasets Used

- **CBIS-DDSM**: Standardized mammogram dataset with ROI annotations and pathology reports.
- **Histopathology IDC**: Whole-slide images divided into IDC-positive and IDC-negative patches.
- **WDBC Dataset**: Structured dataset with 569 samples labeled as benign or malignant.

---

## 📈 Results Summary

| Model                 | Accuracy   | F1 Score  |
|-----------------------|------------|-----------|
| Logistic Regression   | **98.25%** | 0.99 / 0.98 |
| K-Nearest Neighbors   | 97.73%     | 0.98 / 0.97 |
| CNN (Image-Based)     | 93.72%     | -         |
| Decision Tree         | 92.11%     | -         |

- CNN struggled with overfitting due to image variability and dataset size.
- Simpler ML models performed better on structured data due to linear separability and clean features.

---

## 📌 Key Insights

- Combining CNNs and classical ML enables a broader and more robust prediction strategy.
- Structured data benefits more from simpler, interpretable models.
- Image-based models require more data and preprocessing to match structured model performance.

---

## 🚀 Future Work

- Explore more advanced CNN architectures and attention mechanisms.
- Integrate structured and unstructured data for hybrid decision-making.
- Use Explainable AI (XAI) techniques to increase clinical interpretability.
- Train models on larger and more diverse datasets.
- Deploy the model for real-time clinical support.

---

## 👩‍🔬 Authors

- Katikala Dedeepya
- Geshna B
- Malavika S Prasad
- Vada Gouri Hansika Reddy
- Dr. Prajeesh C B (Supervisor)

---

## 📜 License

This project is for educational and research purposes only. Please cite appropriately when using this work in academic settings.

---
