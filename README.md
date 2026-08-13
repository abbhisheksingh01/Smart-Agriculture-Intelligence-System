\# 🌱 Smart Agriculture Intelligence System



An AI-powered agriculture project that uses \*\*Machine Learning and Data Analytics\*\* to analyze agricultural data and predict crop yield.



The main goal of this project is to make agricultural data easier to understand and use for better farming decisions. The system analyzes factors such as crop type, state, cultivation cost, production cost, and historical agricultural information to estimate crop yield.



\---



\## 📌 Project Overview



Agriculture plays an important role in India's economy, but crop production can be affected by many factors such as cultivation cost, crop variety, location, and production conditions.



This project uses machine learning to study available agricultural data and build a model that can predict crop yield.



Along with the machine learning model, a web-based dashboard has been created to present the project in a simple and user-friendly way.



The project combines:



\- 📊 Agricultural Data Analysis

\- 🤖 Machine Learning

\- 🌾 Crop Yield Prediction

\- 📈 Data Visualization

\- 🌐 Web Dashboard

\- 🔍 Model Evaluation

\- 📋 Feature Importance Analysis



\---



\## 🎯 Problem Statement



Crop yield prediction is useful for understanding agricultural production and supporting better planning.



However, agricultural datasets can contain information from different crops and states, making it difficult to identify useful patterns manually.



The problem addressed in this project is:



> \*\*How can machine learning be used to analyze agricultural data and predict crop yield based on crop, location, cultivation cost, and production-related factors?\*\*



\---



\## 💡 Objectives



The main objectives of this project are:



1\. Collect and organize agricultural datasets.

2\. Clean and preprocess the available data.

3\. Perform exploratory data analysis.

4\. Understand relationships between cultivation costs and crop yield.

5\. Train different machine learning regression models.

6\. Compare the performance of multiple models.

7\. Select a suitable final model for crop yield prediction.

8\. Analyze important features affecting prediction.

9\. Create a simple agricultural dashboard.

10\. Provide a practical example of crop yield prediction.



\---



\## 🌾 Key Features



\### 1. Crop Yield Prediction



The system predicts crop yield using agricultural information such as:



\- Crop

\- State

\- Cost of Cultivation (A2+FL)

\- Cost of Cultivation (C2)

\- Cost of Production (C2)



The prediction is given in:



\*\*Quintal/Hectare\*\*



\---



\### 2. Agricultural Data Analysis



The project analyzes agricultural datasets to understand:



\- Crop distribution

\- State-wise agricultural information

\- Average crop yield

\- Production trends

\- Cultivation cost

\- Production cost

\- Relationships between different variables



\---



\### 3. Machine Learning Model Comparison



Several regression models were trained and evaluated:



\- Linear Regression

\- Decision Tree

\- Random Forest

\- Gradient Boosting



Their performance was compared using:



\- MAE

\- RMSE

\- R² Score



\---



\### 4. Feature Importance



Feature importance analysis helps understand which input variables have the greatest influence on the model's predictions.



The final Random Forest model showed that production cost, crop type, and cultivation costs were among the important features.



\---



\### 5. Interactive Web Dashboard



A separate HTML, CSS and JavaScript based dashboard was developed to present the project visually.



The dashboard includes sections for:



\- 🏠 Dashboard

\- 🌱 Crop Prediction

\- 📊 Data Analysis

\- 🤖 Model Performance

\- 🔍 Feature Importance



The design uses an agriculture-inspired green theme to make the application feel more connected to the project domain.



\---



\# 🛠️ Technologies Used



\## Programming Languages



\- Python

\- HTML

\- CSS

\- JavaScript



\## Python Libraries



The machine learning and data analysis part of the project uses Python libraries such as:



\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Seaborn

\- Joblib



\## Machine Learning



\- Linear Regression

\- Decision Tree Regressor

\- Random Forest Regressor

\- Gradient Boosting Regressor



\## Frontend



\- HTML5

\- CSS3

\- JavaScript



\## Development Tools



\- Visual Studio Code

\- Git

\- GitHub

\- Command Prompt / PowerShell



\---



\# 📂 Project Structure



```text

Smart-Agriculture-Intelligence-System/

│

├── data/

│   ├── raw/

│   │   ├── area\_production\_yield.csv

│   │   ├── cost\_yield.csv

│   │   ├── crop\_variety.csv

│   │   ├── production\_history.csv

│   │   └── production\_index.csv

│   │

│   └── processed/

│       ├── area\_production\_yield\_cleaned.csv

│       ├── cost\_yield\_cleaned.csv

│       ├── crop\_variety\_cleaned.csv

│       ├── production\_history\_cleaned.csv

│       └── production\_index\_cleaned.csv

│

├── models/

│   ├── best\_crop\_yield\_model.pkl

│   ├── cross\_validation\_results.csv

│   ├── feature\_importance.csv

│   └── model\_comparison.csv

│

├── outputs/

│   ├── figures/

│   │   ├── actual\_vs\_predicted.png

│   │   ├── average\_yield\_by\_crop.png

│   │   ├── average\_yield\_by\_state.png

│   │   ├── correlation\_heatmap.png

│   │   ├── crop\_distribution.png

│   │   ├── cultivation\_cost\_vs\_yield.png

│   │   ├── feature\_importance.png

│   │   ├── model\_comparison.png

│   │   ├── production\_index\_trend.png

│   │   ├── residual\_plot.png

│   │   └── top\_crop\_production.png

│   │

│   └── prediction\_results.csv

│

├── src/

│   ├── cross\_validation.py

│   ├── data\_analysis.py

│   ├── data\_loader.py

│   ├── eda.py

│   ├── evaluate\_model.py

│   ├── feature\_engineering.py

│   ├── final\_model.py

│   ├── predict.py

│   ├── preprocessing.py

│   └── train\_model.py

│

├── app.py

├── main.py

├── index.html

├── style.css

├── script.js

├── requirements.txt

├── .gitignore

└── README.md

