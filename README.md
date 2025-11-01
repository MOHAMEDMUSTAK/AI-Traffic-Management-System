# AI-Traffic-Management-System
This project demonstrates intelligent signal timing prediction using AI for smart cities.
# 🚦 AI Traffic Management System

An **AI-powered Traffic Management System** that predicts optimal traffic signal times based on real-world data such as traffic volume, weather, and time.  
This project uses machine learning algorithms to analyze traffic patterns and helps in smart signal control for urban cities.

---

## 📊 Dataset
**Dataset used:** `Metro_Interstate_Traffic_Volume.csv`  
Source: Contains hourly traffic volume data along with temperature, rain, and weather conditions.

---

## 🧠 Machine Learning Models Used
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  

Each model is trained and evaluated, and the best-performing one is used for prediction.

---

## ⚙️ Technologies Used
| Component | Technology |
|------------|-------------|
| Programming Language | Python |
| Framework | Flask |
| Libraries | scikit-learn, XGBoost, pandas, matplotlib, seaborn |
| Frontend | HTML, CSS, JavaScript |
| Data Visualization | Line Chart, Bar Chart, Pie Chart, Heatmap, Scatter Plot |

---

## 🧩 Features
✅ Predicts traffic signal timing using AI  
✅ Compares performance of multiple models  
✅ Generates multiple visualizations (bar, pie, heatmap, scatter, etc.)  
✅ Saves charts as PNG automatically  
✅ Flask-based backend with web interface  

---

## 🖼️ Visualizations
The project automatically generates and saves the following plots:
- Bar Chart  
- Pie Chart  
- Correlation Heatmap  
- Traffic Distribution by Weather  
- Traffic vs Temperature  

All saved as `.png` images inside the `graphs/` folder.

---

## 🧰 Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-Traffic-Management-System.git
   cd AI-Traffic-Management-System
