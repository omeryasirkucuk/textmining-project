import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import pickle
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports for deep learning (only if available)
try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    import torch
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    # Create dummy torch module to prevent errors
    class DummyTorch:
        def device(self, *args, **kwargs):
            return "cpu"
        def no_grad(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def is_available(self):
            return False
        @property
        def cuda(self):
            return self
        def __getattr__(self, name):
            # Return a dummy function for any other torch attributes
            return lambda *args, **kwargs: None
    torch = DummyTorch() if not DEEP_LEARNING_AVAILABLE else torch

import warnings
warnings.filterwarnings('ignore')


class FeatureEngineering:
    """
    Advanced feature engineering for recipe classification
    """
    
    def __init__(self):
        self.tfidf_title = TfidfVectorizer(max_features=1000, stop_words=None, ngram_range=(1, 2))
        self.tfidf_ingredients = TfidfVectorizer(max_features=2000, stop_words=None, ngram_range=(1, 2))
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.categorical_columns = None  # Store column names for inference
        
    def create_text_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create TF-IDF features from text"""
        
        # Process title text
        title_text = df['title_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        title_features = self.tfidf_title.fit_transform(title_text)
        
        # Process ingredients text
        ingredients_text = df['ingredients_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        ingredients_features = self.tfidf_ingredients.fit_transform(ingredients_text)
        
        return title_features, ingredients_features
    
    def create_text_features_inference(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create TF-IDF features from text for inference (transform only)"""
        
        # Process title text
        title_text = df['title_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        title_features = self.tfidf_title.transform(title_text)
        
        # Process ingredients text
        ingredients_text = df['ingredients_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        ingredients_features = self.tfidf_ingredients.transform(ingredients_text)
        
        return title_features, ingredients_features
    
    def create_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create and scale numerical features"""
        
        numerical_cols = [
            'calories_per_serving', 'prep_time_minutes', 'cook_time_minutes',
            'total_time_minutes', 'serving_size', 'ingredient_count',
            'healthy_ingredient_total', 'unhealthy_ingredient_total', 'health_ratio',
            'is_fried', 'is_baked', 'is_raw'
        ]
        
        # Add ingredient category features
        ingredient_features = [col for col in df.columns if col.startswith(('healthy_', 'unhealthy_')) and col.endswith('_count')]
        numerical_cols.extend(ingredient_features)
        
        # Select available columns
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        numerical_features = df[available_cols].fillna(0)
        numerical_features_scaled = self.scaler.fit_transform(numerical_features)
        
        return numerical_features_scaled
    
    def create_numerical_features_inference(self, df: pd.DataFrame) -> np.ndarray:
        """Create and scale numerical features for inference (transform only)"""
        
        numerical_cols = [
            'calories_per_serving', 'prep_time_minutes', 'cook_time_minutes',
            'total_time_minutes', 'serving_size', 'ingredient_count',
            'healthy_ingredient_total', 'unhealthy_ingredient_total', 'health_ratio',
            'is_fried', 'is_baked', 'is_raw'
        ]
        
        # Add ingredient category features
        ingredient_features = [col for col in df.columns if col.startswith(('healthy_', 'unhealthy_')) and col.endswith('_count')]
        numerical_cols.extend(ingredient_features)
        
        # Select available columns
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        numerical_features = df[available_cols].fillna(0)
        numerical_features_scaled = self.scaler.transform(numerical_features)
        
        return numerical_features_scaled
    
    def create_categorical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create encoded categorical features"""
        
        # Main category encoding
        main_cat_encoded = pd.get_dummies(df['main_cat'], prefix='cat').astype(int)
        self.categorical_columns = main_cat_encoded.columns.tolist()  # Store for inference
        
        return main_cat_encoded.values
    
    def create_categorical_features_inference(self, df: pd.DataFrame) -> np.ndarray:
        """Create encoded categorical features for inference"""
        
        # Main category encoding
        main_cat_encoded = pd.get_dummies(df['main_cat'], prefix='cat').astype(int)
        
        # Ensure same columns as training
        if self.categorical_columns is not None:
            # Add missing columns with zeros
            for col in self.categorical_columns:
                if col not in main_cat_encoded.columns:
                    main_cat_encoded[col] = 0
            
            # Remove extra columns and reorder
            main_cat_encoded = main_cat_encoded[self.categorical_columns]
        
        return main_cat_encoded.values
    
    def combine_all_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Combine all feature types"""
        
        # Get all feature types
        title_features, ingredients_features = self.create_text_features(df)
        numerical_features = self.create_numerical_features(df)
        categorical_features = self.create_categorical_features(df)
        
        # Combine features
        from scipy.sparse import hstack, csr_matrix
        
        all_features = hstack([
            title_features,
            ingredients_features,
            csr_matrix(numerical_features),
            csr_matrix(categorical_features)
        ])
        
        # Prepare labels
        labels = self.label_encoder.fit_transform(df['health_label_ensemble'])
        
        return all_features, labels
    
    def combine_all_features_inference(self, df: pd.DataFrame) -> np.ndarray:
        """Combine all feature types for inference (transform only)"""
        
        # Get all feature types using inference methods
        title_features, ingredients_features = self.create_text_features_inference(df)
        numerical_features = self.create_numerical_features_inference(df)
        categorical_features = self.create_categorical_features_inference(df)
        
        # Combine features
        from scipy.sparse import hstack, csr_matrix
        
        all_features = hstack([
            title_features,
            ingredients_features,
            csr_matrix(numerical_features),
            csr_matrix(categorical_features)
        ])
        
        return all_features


class TraditionalModels:
    """
    Traditional machine learning models for recipe classification
    """
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineering()
        
    def initialize_models(self):
        """Initialize traditional ML models with regularization to prevent overfitting"""
        
        self.models = {
            # Skip MultinomialNB due to negative values issue
            # 'naive_bayes': MultinomialNB(alpha=1.0),
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=0.1,  # Regularization
                class_weight='balanced'
            ),
            'svm': SVC(
                random_state=42, 
                probability=True,
                C=0.1,  # Regularization
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,  # Limit depth to prevent overfitting
                min_samples_split=20,  # Require more samples to split
                min_samples_leaf=10,   # Require more samples in leaf nodes
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=30,  # Further reduced
                random_state=42,
                max_depth=3,      # More conservative depth
                learning_rate=0.05, # Even more conservative learning rate
                min_samples_split=50,  # Require more samples to split
                min_samples_leaf=25,   # Require more samples in leaf nodes
                subsample=0.7,    # Use only 70% of samples for each tree
                max_features='sqrt'  # Use only sqrt of features per tree
            )
        }
    
    def train_model(self, model_name: str, X_train, y_train, X_val=None, y_val=None):
        """Train a specific model with validation monitoring"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        print(f"Training {model_name}...")
        
        # Special handling for gradient boosting with early stopping
        if model_name == 'gradient_boosting' and X_val is not None and y_val is not None:
            # Train with validation monitoring for early stopping
            model.fit(X_train, y_train)
            
            # Check validation scores during training to detect overfitting
            train_scores = []
            val_scores = []
            
            for i, pred in enumerate(model.staged_predict(X_train)):
                train_score = accuracy_score(y_train, pred)
                train_scores.append(train_score)
                
            for i, pred in enumerate(model.staged_predict(X_val)):
                val_score = accuracy_score(y_val, pred)
                val_scores.append(val_score)
                
            # Find best iteration (where validation score is highest)
            best_iter = np.argmax(val_scores)
            print(f"Best iteration: {best_iter + 1}, Train: {train_scores[best_iter]:.4f}, Val: {val_scores[best_iter]:.4f}")
            
            # Retrain with optimal number of estimators
            if best_iter < len(val_scores) - 5:  # If we stopped early
                model.set_params(n_estimators=best_iter + 1)
                model.fit(X_train, y_train)
                print(f"Retrained with {best_iter + 1} estimators to prevent overfitting")
        else:
            model.fit(X_train, y_train)
        
        # Validation score if validation set provided
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            train_score = model.score(X_train, y_train)
            print(f"{model_name} - Train: {train_score:.4f}, Val: {val_score:.4f}")
            
            # Check for overfitting
            if train_score - val_score > 0.1:
                print(f"⚠️  Warning: Possible overfitting detected (train-val gap: {train_score - val_score:.4f})")
        
        return model
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all traditional models"""
        
        self.initialize_models()
        trained_models = {}
        
        for name in self.models.keys():
            trained_models[name] = self.train_model(name, X_train, y_train, X_val, y_val)
        
        return trained_models
    
    def hyperparameter_tuning(self, model_name: str, X_train, y_train, param_grid: Dict):
        """Perform hyperparameter tuning"""
        
        model = self.models[model_name]
        
        print(f"Tuning hyperparameters for {model_name}...")
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_


class DeepLearningModels:
    """
    Deep learning models using transformers (optional)
    """
    
    def __init__(self):
        if not DEEP_LEARNING_AVAILABLE:
            print("Warning: PyTorch/Transformers not available. Deep learning features disabled.")
            self.tokenizer = None
            self.model = None
            self.device = "cpu"
            return
            
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_turkish_bert(self, model_name='dbmdz/bert-base-turkish-cased'):
        """Load Turkish BERT model"""
        
        if not DEEP_LEARNING_AVAILABLE:
            print("Error: PyTorch/Transformers not available. Cannot load BERT model.")
            return
            
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        
    def create_embeddings(self, texts: List[str], max_length=256) -> np.ndarray:
        """Create BERT embeddings for texts"""
        
        if not DEEP_LEARNING_AVAILABLE:
            print("Error: PyTorch/Transformers not available. Cannot create embeddings.")
            return np.array([])
        
        if self.tokenizer is None or self.model is None:
            self.load_turkish_bert()
        
        embeddings = []
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=max_length,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def create_recipe_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Create embeddings for entire recipes"""
        
        if not DEEP_LEARNING_AVAILABLE:
            print("Error: PyTorch/Transformers not available. Cannot create recipe embeddings.")
            return np.array([])
        
        # Combine title and ingredients
        recipe_texts = []
        for _, row in df.iterrows():
            title = row.get('title', '')
            ingredients = ' '.join(row.get('ingredients', []))
            combined_text = f"{title} {ingredients}"
            recipe_texts.append(combined_text)
        
        print(f"Creating embeddings for {len(recipe_texts)} recipes...")
        embeddings = self.create_embeddings(recipe_texts)
        
        return embeddings


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    def __init__(self, label_encoder=None):
        self.label_encoder = label_encoder
        
    def evaluate_model(self, model, X_test, y_test, model_name: str):
        """Evaluate a single model"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n=== {model_name} Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        if self.label_encoder:
            target_names = self.label_encoder.classes_
        else:
            target_names = None
            
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, model_name, target_names)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name: str, target_names=None):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'models/{model_name}_confusion_matrix.png', dpi=300)
        plt.show()
    
    def compare_models(self, results: Dict[str, Dict]):
        """Compare multiple models"""
        
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        f1_scores = [results[name]['f1_score'] for name in model_names]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        ax1.bar(model_names, accuracies, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # F1-score comparison
        ax2.bar(model_names, f1_scores, color='lightcoral')
        ax2.set_title('Model F1-Score Comparison')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300)
        plt.show()
        
        # Print summary
        print("\n=== Model Comparison Summary ===")
        for name in model_names:
            print(f"{name}: Accuracy={results[name]['accuracy']:.4f}, F1={results[name]['f1_score']:.4f}")


class RecipeClassificationPipeline:
    """
    Complete pipeline for recipe classification
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineering()
        self.traditional_models = TraditionalModels()
        self.dl_models = DeepLearningModels() if DEEP_LEARNING_AVAILABLE else None
        self.evaluator = ModelEvaluator()
        self.best_model = None
        self.best_score = 0
        
    def load_data(self):
        """Load labeled data"""
        
        try:
            df = pd.read_pickle('data/processed/labeled_recipes.pkl')
            print(f"Loaded {len(df)} labeled recipes")
            return df
        except FileNotFoundError:
            print("Error: Labeled data not found. Run labeling.py first.")
            return None
    
    def prepare_data(self, df: pd.DataFrame, test_size=0.2, val_size=0.1):
        """Prepare train/validation/test splits with improved sampling"""
        
        # Filter high-confidence samples for training but lower the threshold
        # to get more data and prevent overfitting
        high_confidence = df[df['label_confidence'] > 0.6].copy()  # Lowered from 0.7
        print(f"Using {len(high_confidence)} high-confidence samples")
        
        # Check class distribution
        label_counts = high_confidence['health_label_ensemble'].value_counts()
        print(f"Label distribution: {dict(label_counts)}")
        
        # Create features and labels
        X, y = self.feature_engineer.combine_all_features(high_confidence)
        
        # Train/test split with stratification
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train/validation split with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size/(1-test_size), 
            random_state=42, stratify=y_train_val
        )
        
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        print(f"Feature dimensions: {X_train.shape[1]} features")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_traditional_models(self, X_train, y_train, X_val, y_val):
        """Train traditional ML models"""
        
        print("\n=== Training Traditional Models ===")
        trained_models = self.traditional_models.train_all_models(
            X_train, y_train, X_val, y_val
        )
        
        return trained_models
    
    def train_deep_models(self, df: pd.DataFrame, y_train, y_val, y_test):
        """Train deep learning models"""
        
        if not DEEP_LEARNING_AVAILABLE or self.dl_models is None:
            print("Deep learning models not available. Skipping...")
            return {}, None
        
        print("\n=== Training Deep Learning Models ===")
        
        # Create embeddings
        embeddings = self.dl_models.create_recipe_embeddings(df)
        
        if embeddings.size == 0:
            print("Failed to create embeddings. Skipping deep learning models.")
            return {}, None
        
        # Split embeddings according to data splits
        # Note: This is simplified - in practice, you'd need to track indices
        X_emb_train = embeddings[:len(y_train)]
        X_emb_val = embeddings[len(y_train):len(y_train)+len(y_val)]
        X_emb_test = embeddings[len(y_train)+len(y_val):]
        
        # Train classifier on embeddings
        rf_emb = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_emb.fit(X_emb_train, y_train)
        
        return {'bert_rf': rf_emb}, X_emb_test
    
    def evaluate_all_models(self, models, X_test, y_test, deep_models=None, X_emb_test=None):
        """Evaluate all trained models"""
        
        self.evaluator.label_encoder = self.feature_engineer.label_encoder
        results = {}
        
        # Evaluate traditional models
        print("\n=== Evaluating Traditional Models ===")
        for name, model in models.items():
            results[name] = self.evaluator.evaluate_model(model, X_test, y_test, name)
            
            # Track best model (exclude gradient boosting if it shows signs of overfitting)
            if results[name]['f1_score'] > self.best_score:
                # If gradient boosting has suspiciously high performance, prefer other models
                if name == 'gradient_boosting' and results[name]['f1_score'] > 0.98:
                    print(f"⚠️  Gradient boosting shows signs of overfitting ({results[name]['f1_score']:.4f}), considering other models")
                    continue
                self.best_score = results[name]['f1_score']
                self.best_model = model
        
        # Evaluate deep models
        if deep_models and X_emb_test is not None:
            print("\n=== Evaluating Deep Learning Models ===")
            for name, model in deep_models.items():
                results[name] = self.evaluator.evaluate_model(model, X_emb_test, y_test, name)
                
                if results[name]['f1_score'] > self.best_score:
                    self.best_score = results[name]['f1_score']
                    self.best_model = model
        
        # Compare all models
        self.evaluator.compare_models(results)
        
        return results
    
    def save_best_model(self):
        """Save the best performing model"""
        
        if self.best_model is not None:
            model_path = 'models/best_recipe_classifier.pkl'
            joblib.dump(self.best_model, model_path)
            
            # Save feature engineering components
            feature_path = 'models/feature_engineering.pkl'
            joblib.dump(self.feature_engineer, feature_path)
            
            print(f"\nBest model saved to {model_path}")
            print(f"Feature engineering saved to {feature_path}")
            print(f"Best F1-Score: {self.best_score:.4f}")
    
    def run_complete_pipeline(self):
        """Run the complete model training and evaluation pipeline"""
        
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df)
        
        # Train traditional models
        traditional_models = self.train_traditional_models(X_train, y_train, X_val, y_val)
        
        # Train deep learning models (optional)
        # deep_models, X_emb_test = self.train_deep_models(df, y_train, y_val, y_test)
        
        # Evaluate all models
        results = self.evaluate_all_models(traditional_models, X_test, y_test)
        
        # Save best model
        self.save_best_model()
        
        return results


def main():
    """Main model training pipeline"""
    
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    
    # Run complete pipeline
    pipeline = RecipeClassificationPipeline()
    results = pipeline.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    results = main() 