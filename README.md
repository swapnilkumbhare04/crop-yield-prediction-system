# Crop Yield Prediction System

**Author:** Swapnil Sanjay Kumbhare

**Institution:** Rajiv Gandhi College of Engineering, Research & Technology, Chandrapur  
**Program:** B.Tech, Computer Science & Engineering  
**Project:** AICTE MS Elevate Internship - January 2026

## 📋 Project Overview

The Crop Yield Prediction System is an interactive web application (built with Streamlit) and Jupyter Notebook that uses machine learning algorithms to forecast agricultural output. By inputting farm conditions such as soil type, rainfall, temperature, humidity, and fertilizer usage, users can obtain accurate yield predictions for various crops.

## 🚀 Features

- **Crop Yield Prediction**: Predict yield in tonnes per hectare for 10 major crops
- **Interactive User Interface**: User-friendly web interface (Streamlit) or ipywidgets-based form
- **Crop Recommendation**: Suggests suitable crops based on environmental conditions
- **Revenue Estimation**: Calculates expected revenue based on current market rates
- **Confidence Range**: Provides prediction intervals for better decision-making

### Supported Crops
- Bajra, Barley, Cotton, Groundnut, Jowar, Maize, Rice, Soybean, Sugarcane, Wheat

### Supported Soil Types
- Alluvial, Arid, Black, Forest, Laterite, Red

## 📁 File Structure

```
crop_yield_project/
│
├── crop_yield_project.ipynb    # Main Jupyter Notebook with ML model and UI
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── data/
│   └── crop_yield.csv         # Dataset used for training the model
├── models/
│   └── crop_yield_model.pkl   # Trained Random Forest model
└── README.md                  # Project documentation
```

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python 3.x |
| Framework | Jupyter Notebook, Streamlit |
| UI Components | ipywidgets, Streamlit widgets |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib |

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Streamlit

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd crop_yield_project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
```

**Activate on Windows:**
```bash
venv\Scripts\activate
```

**Activate on macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## ▶️ How to Run

### Method 1: Streamlit Web Application (Recommended)
```bash
streamlit run app.py
```
The application will open in your default web browser at `http://localhost:8501`

### Method 2: Jupyter Notebook
```bash
jupyter notebook crop_yield_project.ipynb
```
Then press `Shift + Enter` to run cells sequentially, or click "Run" → "Run All Cells" from the menu.

### Method 3: Jupyter Lab
```bash
jupyter lab crop_yield_project.ipynb
```

### Method 4: VS Code
1. Open the project in VS Code
2. Install the Jupyter extension
3. Open `crop_yield_project.ipynb`
4. Click "Run All Cells" or run cells individually

## 📖 User Guide

### Using the Prediction System

1. **Select Crop**: Choose the crop you want to grow from the dropdown menu
2. **Select Soil Type**: Choose your soil type from available options
3. **Enter Environmental Conditions**:
   - Rainfall (mm): Average annual rainfall
   - Temperature (°C): Average temperature
   - Humidity (%): Relative humidity level
   - Soil pH: Soil acidity/alkalinity level
4. **Fertilizer Input** (Optional):
   - N (kg/ha): Nitrogen dosage
   - P (kg/ha): Phosphorus dosage
   - K (kg/ha): Potassium dosage
5. **Click "Predict Yield"**: View predicted yield, confidence range, and revenue estimate

### Using Crop Recommendations

Click "Get Crop Recommendations" to receive personalized crop suggestions based on your current environmental conditions.

## 📊 Sample Prediction Output

```
Prediction Results for Jowar:
├── Predicted Yield: 4.1 tonnes/hectare
├── Confidence Range: 3.48 - 4.71 tonnes/hectare
└── Expected Revenue: ₹61,500 per hectare
```

## ⚙️ Configuration

The model uses the following default parameters:
- **Algorithm**: Random Forest Regressor
- **Test Size**: 20% of dataset
- **Random State**: 42 (for reproducibility)

## 🔧 Customization

### Adding New Crops
To add support for additional crops, update the crop list in the dropdown widget configuration section of the notebook.

### Modifying Prediction Model
The machine learning model can be modified in the model training section. Available options include:
- Random Forest
- Decision Tree
- Gradient Boosting
- Linear Regression

## 📝 Notes

- Ensure all dependencies are installed before running the notebook
- The prediction model is trained on synthetic data and should be used for educational/demonstration purposes
- Actual yields may vary based on pests, diseases, and local conditions
- Market prices are illustrative and should be verified locally

## 📄 License

This project is submitted as part of an internship submission.

## 👤 Author

**Swapnil Sanjay Kumbhare**  
B.Tech CSE, Rajiv Gandhi College of Engineering, Research & Technology, Chandrapur

---

For technical support or queries, please refer to the notebook comments or contact the project administrator.
