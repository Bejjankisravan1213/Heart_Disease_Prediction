https://github.com/Bejjankisravan1213/Heart_Disease_Prediction

# Heart Disease Prediction

## 🏥 Project Overview
This is a **Machine Learning project** that predicts whether a person is at risk of heart disease based on medical features like age, cholesterol, blood pressure, chest pain type, and more. The model helps in early detection and awareness for heart-related conditions.

Contents:
- heart.csv
- heart_prediction.ipynb
- model.pkl
- images/


---

## 📊 Dataset
- Source: UCI Heart Disease dataset  
- 303 rows, 14 features including:
  - age, sex, cp (chest pain type), trestbps (resting BP), chol (cholesterol)
  - fbs (fasting blood sugar), restecg (ECG results), thalach (max heart rate)
  - exang (exercise induced angina), oldpeak, slope, ca (vessels), thal, target
- `target` column: 1 = Heart Disease, 0 = No Heart Disease

---

## ⚙️ Tech Stack
- Python 3.x  
- Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Pickle  

---

## 🛠 Features
- Logistic Regression model for prediction  
- Data visualization: Count plots, correlation heatmaps  
- Train/test split with feature scaling  
- Saved model using `pickle` for future predictions  

---

## 📈 How to Run
1. Clone this repository:
git clone https://github.com/yourusername/Heart-Disease-Prediction.git

cd Heart-Disease-Prediction

2. Install required libraries:


pip install numpy pandas scikit-learn matplotlib seaborn

3. Run the script:


python heart_prediction.py

4. The trained model will be saved as `model.pkl`.

---

## 🧪 Testing the Model
You can test predictions using:

```python
import pickle

model = pickle.load(open("model.pkl", "rb"))

# Example input: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
sample = [[63,1,1,145,233,1,0,150,0,2.3,0,0,1]]

prediction = model.predict(sample)
print("Result:", "Heart Disease" if prediction[0]==1 else "No Disease")

📌 Results

Accuracy: ~80–88% (varies depending on dataset split)

Confusion matrix and classification report provided in the script


📝 Future Improvements:

Implement advanced models like Random Forest, XGBoost, SVM

Deploy as a Web App using Flask or Streamlit

Add more visualizations for EDA

Improve dataset with more patient records
