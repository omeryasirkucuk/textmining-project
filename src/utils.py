import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple
import json
import pickle
import os
from wordcloud import WordCloud
from collections import Counter

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataUtils:
    """
    Utility functions for data operations
    """
    
    @staticmethod
    def load_json_data(file_path: str) -> List[Dict]:
        """Load JSON data safely"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} records from {file_path}")
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return []
    
    @staticmethod
    def save_data(data: Any, file_path: str, format_type: str = 'pickle'):
        """Save data in specified format"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format_type == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        elif format_type == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format_type == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            else:
                raise ValueError("CSV format requires DataFrame input")
        
        print(f"Data saved to {file_path}")
    
    @staticmethod
    def load_pickle(file_path: str) -> Any:
        """Load pickle file safely"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
    
    @staticmethod
    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive information about DataFrame"""
        info = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'null_counts': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        return info


class VisualizationUtils:
    """
    Utility functions for creating visualizations
    """
    
    @staticmethod
    def plot_label_distribution(labels: pd.Series, title: str = "Label Distribution"):
        """Plot distribution of labels"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        labels.value_counts().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title(f'{title} - Counts')
        ax1.set_xlabel('Labels')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        labels.value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title(f'{title} - Percentages')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_calorie_distribution(calories: pd.Series, bins: int = 50):
        """Plot calorie distribution with health categories"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(calories[calories > 0], bins=bins, alpha=0.7, color='lightcoral')
        ax1.axvline(200, color='green', linestyle='--', label='Healthy threshold (≤200)')
        ax1.axvline(400, color='orange', linestyle='--', label='Moderate threshold (≤400)')
        ax1.set_title('Calorie Distribution')
        ax1.set_xlabel('Calories per Serving')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Box plot with categories
        calorie_categories = pd.cut(calories[calories > 0], 
                                  bins=[0, 200, 400, float('inf')], 
                                  labels=['Healthy', 'Moderate', 'FastFood'])
        
        data_for_box = pd.DataFrame({'Calories': calories[calories > 0], 
                                   'Category': calorie_categories})
        
        sns.boxplot(data=data_for_box, x='Category', y='Calories', ax=ax2)
        ax2.set_title('Calories by Health Category')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_ingredient_wordcloud(ingredients_list: List[List[str]], 
                                  title: str = "Most Common Ingredients"):
        """Create word cloud from ingredients"""
        
        # Flatten ingredients and count
        all_ingredients = []
        for ingredient_list in ingredients_list:
            all_ingredients.extend(ingredient_list)
        
        ingredient_text = ' '.join(all_ingredients)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(ingredient_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_correlation(df: pd.DataFrame, features: List[str]):
        """Plot correlation matrix for selected features"""
        
        # Select numeric features only
        numeric_features = df[features].select_dtypes(include=[np.number])
        
        # Calculate correlation
        correlation_matrix = numeric_features.corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_category_health_distribution(df: pd.DataFrame, category_col: str, 
                                        health_col: str, top_n: int = 10):
        """Plot health distribution across recipe categories"""
        
        # Get top categories
        top_categories = df[category_col].value_counts().head(top_n).index
        filtered_df = df[df[category_col].isin(top_categories)]
        
        # Create cross-tabulation
        ct = pd.crosstab(filtered_df[category_col], filtered_df[health_col], normalize='index')
        
        # Plot stacked bar chart
        ax = ct.plot(kind='bar', stacked=True, figsize=(12, 8), 
                    color=['green', 'orange', 'red'])
        plt.title(f'Health Distribution by {category_col.title()}')
        plt.xlabel(category_col.title())
        plt.ylabel('Proportion')
        plt.legend(title='Health Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


class EvaluationUtils:
    """
    Utility functions for model evaluation
    """
    
    @staticmethod
    def calculate_agreement_metrics(labels1: pd.Series, labels2: pd.Series) -> Dict[str, float]:
        """Calculate agreement metrics between two label sets"""
        
        # Ensure same length
        min_len = min(len(labels1), len(labels2))
        labels1 = labels1.iloc[:min_len]
        labels2 = labels2.iloc[:min_len]
        
        # Remove NaN values
        mask = labels1.notna() & labels2.notna()
        labels1 = labels1[mask]
        labels2 = labels2[mask]
        
        # Calculate metrics
        agreement = (labels1 == labels2).mean()
        
        # Cohen's Kappa (simplified)
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(labels1, labels2)
        
        return {
            'agreement': agreement,
            'kappa': kappa,
            'total_compared': len(labels1)
        }
    
    @staticmethod
    def create_confusion_matrix_plot(y_true, y_pred, labels=None, title="Confusion Matrix"):
        """Create a beautiful confusion matrix plot"""
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return cm
    
    @staticmethod
    def plot_model_comparison(results: Dict[str, Dict[str, float]], 
                            metrics: List[str] = ['accuracy', 'f1_score']):
        """Plot comparison of multiple models"""
        
        models = list(results.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            
            bars = axes[i].bar(models, values, color='skyblue', alpha=0.7)
            axes[i].set_title(f'Model Comparison - {metric.title()}')
            axes[i].set_ylabel(metric.title())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()


class TextAnalysisUtils:
    """
    Utility functions for text analysis
    """
    
    @staticmethod
    def analyze_text_features(texts: pd.Series) -> Dict[str, Any]:
        """Analyze text features like length, word count, etc."""
        
        # Remove NaN values
        texts = texts.dropna()
        
        # Calculate features
        text_lengths = texts.str.len()
        word_counts = texts.str.split().str.len()
        
        analysis = {
            'avg_length': text_lengths.mean(),
            'median_length': text_lengths.median(),
            'max_length': text_lengths.max(),
            'min_length': text_lengths.min(),
            'avg_words': word_counts.mean(),
            'median_words': word_counts.median(),
            'max_words': word_counts.max(),
            'min_words': word_counts.min()
        }
        
        return analysis
    
    @staticmethod
    def extract_common_words(texts: pd.Series, top_n: int = 20) -> List[Tuple[str, int]]:
        """Extract most common words from text series"""
        
        # Combine all texts
        all_text = ' '.join(texts.dropna().astype(str))
        
        # Simple word extraction (can be improved with proper tokenization)
        words = all_text.lower().split()
        
        # Count words
        word_counts = Counter(words)
        
        return word_counts.most_common(top_n)
    
    @staticmethod
    def plot_text_length_distribution(texts: pd.Series, title: str = "Text Length Distribution"):
        """Plot distribution of text lengths"""
        
        lengths = texts.dropna().str.len()
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(lengths, bins=30, alpha=0.7, color='lightblue')
        plt.title(f'{title} - Character Count')
        plt.xlabel('Characters')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        word_counts = texts.dropna().str.split().str.len()
        plt.hist(word_counts, bins=30, alpha=0.7, color='lightgreen')
        plt.title(f'{title} - Word Count')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()


def create_project_summary(data_path: str = 'data/processed/') -> Dict[str, Any]:
    """Create a comprehensive project summary"""
    
    summary = {
        'timestamp': pd.Timestamp.now(),
        'data_files': [],
        'model_files': [],
        'statistics': {}
    }
    
    # Check for data files
    data_files = ['processed_recipes.pkl', 'labeled_recipes.pkl', 'sample_for_review.csv']
    for file in data_files:
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            summary['data_files'].append(file)
            
            if file.endswith('.pkl'):
                try:
                    df = pd.read_pickle(file_path)
                    summary['statistics'][file] = {
                        'shape': df.shape,
                        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
                    }
                except:
                    pass
    
    # Check for model files
    model_dir = 'models/'
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith(('.pkl', '.png'))]
        summary['model_files'] = model_files
    
    return summary


def print_project_status():
    """Print current project status"""
    
    print("=" * 60)
    print("TURKISH RECIPE HEALTH CLASSIFIER - PROJECT STATUS")
    print("=" * 60)
    
    # Check data files
    print("\n📁 DATA FILES:")
    data_files = {
        'Raw Data': 'data/detailed_recipe_categorie_unitsize_calorie_chef.json',
        'Processed Data': 'data/processed/processed_recipes.pkl',
        'Labeled Data': 'data/processed/labeled_recipes.pkl',
        'Review Sample': 'data/processed/sample_for_review.csv'
    }
    
    for name, path in data_files.items():
        status = "✅" if os.path.exists(path) else "❌"
        print(f"  {status} {name}: {path}")
    
    # Check source files
    print("\n🐍 SOURCE FILES:")
    src_files = {
        'Data Preprocessing': 'src/data_preprocessing.py',
        'Labeling': 'src/labeling.py',
        'Models': 'src/models.py',
        'Utils': 'src/utils.py'
    }
    
    for name, path in src_files.items():
        status = "✅" if os.path.exists(path) else "❌"
        print(f"  {status} {name}: {path}")
    
    # Check model files
    print("\n🤖 MODEL FILES:")
    model_dir = 'models/'
    if os.path.exists(model_dir):
        model_files = os.listdir(model_dir)
        if model_files:
            for file in model_files:
                print(f"  ✅ {file}")
        else:
            print("  ❌ No model files found")
    else:
        print("  ❌ Models directory not found")
    
    # Check web app
    print("\n🌐 WEB APPLICATION:")
    app_status = "✅" if os.path.exists('streamlit_app.py') else "❌"
    print(f"  {app_status} Streamlit App: streamlit_app.py")
    
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS:")
    
    if not os.path.exists('data/processed/processed_recipes.pkl'):
        print("  1. Run data preprocessing: python src/data_preprocessing.py")
    elif not os.path.exists('data/processed/labeled_recipes.pkl'):
        print("  1. Run labeling: python src/labeling.py")
    elif not os.path.exists('models/best_recipe_classifier.pkl'):
        print("  1. Train models: python src/models.py")
    else:
        print("  1. Launch web app: streamlit run streamlit_app.py")
        print("  2. All systems ready! 🚀")
    
    print("=" * 60)


if __name__ == "__main__":
    print_project_status() 