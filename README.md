# Turkish Recipe Health Classification System

A machine learning-powered web application that classifies Turkish recipes into health categories: **Healthy**, **Moderately Healthy**, and **FastFood**.

## 🎯 Features

- **Interactive Web Interface**: Streamlit-based app for easy recipe classification
- **Multiple Classification Methods**: Rule-based, heuristic, and ML model approaches
- **Dataset Explorer**: Browse and analyze 25,512 Turkish recipes
- **Real-time Classification**: Instant health assessment of recipes
- **Performance Metrics**: View model performance and comparisons

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run streamlit_app.py
```

### 3. Open in Browser
Visit `http://localhost:8501` to use the application.

> **Note**: The application comes with a pre-collected dataset of 25,512 Turkish recipes scraped from yemek.com. The scraping script (`scrapping.py`) is included for reference and transparency.

## 📊 Dataset

- **Source**: 25,512 Turkish recipes from yemek.com
- **Training Data**: 14,313 high-confidence labeled samples
- **Features**: Recipe titles, ingredients, calories, cooking times, categories
- **Labels**: Healthy, Moderately Healthy, FastFood

### 🔍 Data Collection Process

The dataset was collected through web scraping from **yemek.com**, one of Turkey's largest recipe platforms. The data collection process involved:

**1. Recipe URL Collection**
- Systematically crawled recipe category pages
- Extracted individual recipe URLs across all major categories
- Collected 25,512+ unique recipe URLs

**2. Detailed Recipe Scraping** (`scrapping.py`)
- **Target Site**: yemek.com
- **Method**: BeautifulSoup + requests with proper headers
- **Extracted Fields**:
  - Recipe title and unique ID
  - Category hierarchy (main and sub-categories)
  - Serving size (portion information)
  - Preparation and cooking times
  - Complete ingredient lists with quantities
  - Recipe images and URLs

**3. Data Structure**
```json
{
  "id": "unique_recipe_id",
  "url": "original_recipe_url", 
  "title": "Recipe Name",
  "categories": ["CATEGORY", "SUBCATEGORY"],
  "size": "4 kişilik",
  "preparing_time": "15 dakika",
  "cooking_time": "30 dakika", 
  "ingredients": ["2 adet domates", "1 su bardağı bulgur", ...]
}
```

**4. Data Quality & Coverage**
- **Categories**: 15+ main recipe categories (Soups, Salads, Main Dishes, Desserts, etc.)
- **Geographic Coverage**: Traditional Turkish cuisine from all regions
- **Recipe Complexity**: From simple salads to complex main courses
- **Time Range**: Recipes collected represent contemporary Turkish cooking

**5. Ethical Considerations**
- Respectful scraping with appropriate delays
- User-Agent headers to identify scraping activity
- No overloading of the target server
- Data used for academic research purposes

## 🤖 Model Performance

The system uses an ensemble of approaches:

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 93.99% | 93.99% |
| SVM | 93.22% | 93.22% |
| Random Forest | 92.50% | 92.50% |
| Gradient Boosting | 77.81% | 77.81% |

**Best Model**: Logistic Regression (currently deployed)

## 🏗️ Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── run_app.py               # Simple application launcher
├── scrapping.py             # Web scraping script for yemek.com
├── src/                      # Core modules
│   ├── data_preprocessing.py # Data processing and feature extraction
│   ├── labeling.py          # Rule-based and heuristic labeling
│   ├── models.py            # ML models and training pipeline
│   └── utils.py             # Utility functions
├── models/                   # Trained models and visualizations
│   ├── best_recipe_classifier.pkl
│   ├── feature_engineering.pkl
│   └── *.png                # Model performance visualizations
├── data/                     # Dataset and processed files
│   ├── detailed_recipe_categorie_unitsize_calorie_chef.json
│   └── processed/           # Preprocessed data
└── requirements.txt          # Python dependencies
```

## 🍽️ Health Classification Criteria

### 🟢 Healthy
- ≤ 200 calories per serving
- Rich in vegetables and fruits
- Minimal processed ingredients
- Healthy cooking methods

### 🟡 Moderately Healthy
- 200-400 calories per serving
- Balanced macronutrients
- Some processed ingredients
- Moderate cooking methods

### 🔴 FastFood
- > 400 calories per serving
- High in processed ingredients
- Deep-fried or heavy oil usage
- Fast-food style ingredients

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML**: Scikit-learn
- **NLP**: NLTK, Turkish text processing
- **Visualization**: Plotly, Matplotlib
- **Data**: Pandas, NumPy

## 📝 Usage Examples

### Manual Recipe Entry
1. Go to "Recipe Classifier" tab
2. Select "Manual Entry"
3. Fill in recipe details (title, ingredients, calories, etc.)
4. Click "Classify Recipe"

### Browse Dataset
1. Go to "Recipe Classifier" tab
2. Select "Select from Dataset"
3. Search or browse by category
4. Select a recipe to classify

### View Performance
1. Go to "Model Performance" tab
2. View training metrics and model comparisons
3. See feature importance analysis

## 🎓 Academic Context

This project was developed as part of a Text Mining course, focusing on:
- Turkish language processing
- Health classification of traditional recipes
- Ensemble learning approaches
- Real-world application deployment

## 📄 License

This project is for academic purposes.

---

**Note**: The application requires the trained models in the `models/` directory and the dataset in the `data/` directory to function properly. 