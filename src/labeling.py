import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import random
from dataclasses import dataclass
import os
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class HealthCriteria:
    """Define health classification criteria"""
    
    # Calorie thresholds per serving
    HEALTHY_CALORIES_MAX = 200
    MODERATE_CALORIES_MAX = 400
    
    # Healthy ingredient minimum counts
    HEALTHY_VEGETABLE_MIN = 2
    HEALTHY_PROTEIN_MIN = 1
    
    # Unhealthy ingredient maximum counts
    UNHEALTHY_PROCESSED_MAX = 1
    UNHEALTHY_FRIED_MAX = 0
    
    # Health ratio thresholds
    HEALTHY_RATIO_MIN = 0.6
    MODERATE_RATIO_MIN = 0.3


class RuleBasedLabeler:
    """
    Rule-based labeling using defined health criteria
    """
    
    def __init__(self, criteria: HealthCriteria = None):
        self.criteria = criteria or HealthCriteria()
    
    def label_recipe(self, recipe_row: pd.Series) -> str:
        """Label a single recipe based on rules"""
        
        # Extract features
        calories_per_serving = recipe_row.get('calories_per_serving', 0)
        health_ratio = recipe_row.get('health_ratio', 0)
        healthy_total = recipe_row.get('healthy_ingredient_total', 0)
        unhealthy_total = recipe_row.get('unhealthy_ingredient_total', 0)
        is_fried = recipe_row.get('is_fried', 0)
        
        # Apply rules for FastFood
        if (calories_per_serving > self.criteria.MODERATE_CALORIES_MAX or
            is_fried > 0 or
            unhealthy_total >= 3 or
            health_ratio < 0.2):
            return 'FastFood'
        
        # Get category for special rules
        main_cat = recipe_row.get('main_cat', '').lower()
        
        # Apply rules for Healthy (more lenient for vegetables)
        healthy_threshold = 1 if 'sebze' in main_cat else 2
        calorie_threshold = 250 if 'sebze' in main_cat else self.criteria.HEALTHY_CALORIES_MAX
        
        if (calories_per_serving <= calorie_threshold and
            health_ratio >= self.criteria.HEALTHY_RATIO_MIN and
            healthy_total >= healthy_threshold and
            unhealthy_total <= 1):
            return 'Healthy'
        
        # Default to Moderately Healthy
        return 'Moderately Healthy'
    
    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label entire dataset"""
        df = df.copy()
        df['health_label_rule'] = df.apply(self.label_recipe, axis=1)
        return df


class HeuristicLabeler:
    """
    Heuristic labeling using ingredient keywords and patterns
    """
    
    def __init__(self):
        self.setup_heuristic_keywords()
    
    def setup_heuristic_keywords(self):
        """Setup keyword patterns for heuristic classification"""
        
        self.strong_healthy_indicators = [
            'çiğ', 'salata', 'haşlama', 'ızgara', 'buğulama', 'çorba',
            'sebze', 'meyve', 'yoğurt', 'balık', 'tavuk göğsü', 'zeytinyağı'
        ]
        
        self.strong_unhealthy_indicators = [
            'kızartma', 'fritür', 'derin yağ', 'tereyağı', 'krema', 'şeker',
            'çikolata', 'pasta', 'börek', 'sosis', 'sucuk', 'pastırma'
        ]
        
        self.moderate_indicators = [
            'fırın', 'pişirme', 'haşlama', 'et', 'pilav', 'makarna', 'ekmek'
        ]
    
    def calculate_heuristic_score(self, recipe_row: pd.Series) -> float:
        """Calculate heuristic health score"""
        
        title = recipe_row.get('title', '').lower()
        ingredients_text = recipe_row.get('ingredients_text', '').lower()
        main_cat = recipe_row.get('main_cat', '').lower()
        
        full_text = f"{title} {ingredients_text} {main_cat}"
        
        score = 0.5  # neutral start
        
        # Check for strong healthy indicators
        for indicator in self.strong_healthy_indicators:
            if indicator in full_text:
                score += 0.15
        
        # Check for strong unhealthy indicators
        for indicator in self.strong_unhealthy_indicators:
            if indicator in full_text:
                score -= 0.2
        
        # Category-based adjustments
        if any(cat in main_cat for cat in ['salata', 'çorba', 'zeytinyağlı', 'sebze']):
            score += 0.3  # Strong boost for vegetable recipes
        elif any(cat in main_cat for cat in ['tatlı', 'börek', 'kızartma']):
            score -= 0.25
        
        return max(0, min(1, score))  # Clamp between 0 and 1
    
    def label_recipe_heuristic(self, recipe_row: pd.Series) -> str:
        """Label recipe using heuristic approach"""
        
        score = self.calculate_heuristic_score(recipe_row)
        calories_per_serving = recipe_row.get('calories_per_serving', 0)
        
        # Adjust score based on calories
        if calories_per_serving > 500:
            score -= 0.3
        elif calories_per_serving < 150:
            score += 0.2
        
        # Final classification
        if score >= 0.7:
            return 'Healthy'
        elif score <= 0.3:
            return 'FastFood'
        else:
            return 'Moderately Healthy'
    
    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label entire dataset with heuristic approach"""
        df = df.copy()
        df['health_score_heuristic'] = df.apply(self.calculate_heuristic_score, axis=1)
        df['health_label_heuristic'] = df.apply(self.label_recipe_heuristic, axis=1)
        return df


class LLMLabeler:
    """
    LLM-based labeling using OpenAI or similar models
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            if not OPENAI_AVAILABLE:
                print("Warning: OpenAI package not installed. LLM labeling will be skipped.")
            else:
                print("Warning: No OpenAI API key provided. LLM labeling will be skipped.")
    
    def create_prompt(self, recipe_data: Dict[str, Any]) -> str:
        """Create prompt for LLM classification"""
        
        prompt = f"""
Türkçe yemek tarifini sağlık durumuna göre sınıflandır. Sadece şu kategorilerden birini seç: "Healthy", "Moderately Healthy", "FastFood"

Kriter:
- Healthy: ≤200 kalori/porsiyon, doğal malzemeler, az işlenmiş, sebze/meyve ağırlıklı
- Moderately Healthy: 200-400 kalori/porsiyon, dengeli malzemeler, orta düzey işlenmiş
- FastFood: >400 kalori/porsiyon, yüksek işlenmiş malzemeler, kızartma/yağlı

Tarif Bilgileri:
Başlık: {recipe_data.get('title', 'N/A')}
Kategori: {recipe_data.get('main_cat', 'N/A')}
Kalori: {recipe_data.get('calorie', 'N/A')}
Porsiyon: {recipe_data.get('size', 'N/A')}
Malzemeler: {', '.join(recipe_data.get('ingredients', [])[:10])}

Sadece kategori adını yaz, açıklama yapma:"""
        
        return prompt
    
    def label_recipe_llm(self, recipe_data: Dict[str, Any]) -> str:
        """Label single recipe using LLM"""
        
        if not self.client:
            return 'Moderately Healthy'  # Default fallback
        
        try:
            prompt = self.create_prompt(recipe_data)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sen Türkçe yemek tariflerini sağlık durumuna göre sınıflandıran bir uzmansın."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            label = response.choices[0].message.content.strip()
            
            # Validate response
            if label in ['Healthy', 'Moderately Healthy', 'FastFood']:
                return label
            else:
                return 'Moderately Healthy'  # Default fallback
                
        except Exception as e:
            print(f"LLM labeling error: {e}")
            return 'Moderately Healthy'  # Default fallback
    
    def label_sample(self, df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
        """Label a sample of recipes using LLM"""
        
        if not self.client:
            print("Skipping LLM labeling - no API key")
            return df
        
        # Sample recipes
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
        
        print(f"Labeling {len(sample_df)} recipes with LLM...")
        
        llm_labels = []
        for idx, row in sample_df.iterrows():
            recipe_data = row.to_dict()
            label = self.label_recipe_llm(recipe_data)
            llm_labels.append(label)
            
            if len(llm_labels) % 10 == 0:
                print(f"Processed {len(llm_labels)} recipes...")
        
        sample_df['health_label_llm'] = llm_labels
        
        # Merge back to main dataframe
        df = df.merge(
            sample_df[['id', 'health_label_llm']], 
            on='id', 
            how='left'
        )
        
        return df


class EnsembleLabeler:
    """
    Combine multiple labeling approaches for robust classification
    """
    
    def __init__(self):
        self.rule_labeler = RuleBasedLabeler()
        self.heuristic_labeler = HeuristicLabeler()
        self.llm_labeler = LLMLabeler()
    
    def create_ensemble_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ensemble labels from multiple approaches"""
        
        df = df.copy()
        
        # Apply all labeling methods
        print("Applying rule-based labeling...")
        df = self.rule_labeler.label_dataset(df)
        
        print("Applying heuristic labeling...")
        df = self.heuristic_labeler.label_dataset(df)
        
        print("Applying LLM labeling (sample)...")
        df = self.llm_labeler.label_sample(df, sample_size=500)
        
        # Create ensemble label
        df['health_label_ensemble'] = df.apply(self._ensemble_vote, axis=1)
        
        return df
    
    def _ensemble_vote(self, row: pd.Series) -> str:
        """Ensemble voting for final label"""
        
        labels = []
        
        # Rule-based label (weight: 1)
        if 'health_label_rule' in row:
            labels.append(row['health_label_rule'])
        
        # Heuristic label (weight: 1)
        if 'health_label_heuristic' in row:
            labels.append(row['health_label_heuristic'])
        
        # LLM label (weight: 2, if available)
        if 'health_label_llm' in row and pd.notna(row['health_label_llm']):
            labels.extend([row['health_label_llm']] * 2)
        
        if not labels:
            return 'Moderately Healthy'
        
        # Majority vote
        label_counts = pd.Series(labels).value_counts()
        return label_counts.index[0]
    
    def create_confidence_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create confidence scores for labels"""
        
        df = df.copy()
        
        def calculate_confidence(row):
            labels = []
            
            if 'health_label_rule' in row:
                labels.append(row['health_label_rule'])
            if 'health_label_heuristic' in row:
                labels.append(row['health_label_heuristic'])
            if 'health_label_llm' in row and pd.notna(row['health_label_llm']):
                labels.append(row['health_label_llm'])
            
            if not labels:
                return 0.5
            
            # Calculate agreement ratio
            ensemble_label = row.get('health_label_ensemble', 'Moderately Healthy')
            agreement_count = sum(1 for label in labels if label == ensemble_label)
            return agreement_count / len(labels)
        
        df['label_confidence'] = df.apply(calculate_confidence, axis=1)
        
        return df


def main():
    """Main labeling pipeline"""
    
    # Load processed data
    try:
        df = pd.read_pickle('data/processed/processed_recipes.pkl')
        print(f"Loaded {len(df)} processed recipes")
    except FileNotFoundError:
        print("Error: Processed data not found. Run data_preprocessing.py first.")
        return
    
    # Initialize ensemble labeler
    ensemble = EnsembleLabeler()
    
    # Create labels
    df = ensemble.create_ensemble_labels(df)
    
    # Calculate confidence scores
    df = ensemble.create_confidence_scores(df)
    
    # Display results
    print("\n=== Labeling Results ===")
    
    if 'health_label_rule' in df.columns:
        print("\nRule-based distribution:")
        print(df['health_label_rule'].value_counts())
    
    if 'health_label_heuristic' in df.columns:
        print("\nHeuristic distribution:")
        print(df['health_label_heuristic'].value_counts())
    
    if 'health_label_ensemble' in df.columns:
        print("\nEnsemble distribution:")
        print(df['health_label_ensemble'].value_counts())
        
        print(f"\nAverage confidence: {df['label_confidence'].mean():.3f}")
        print(f"High confidence samples (>0.8): {(df['label_confidence'] > 0.8).sum()}")
    
    # Save labeled data
    df.to_pickle('data/processed/labeled_recipes.pkl')
    print("\nLabeled data saved to data/processed/labeled_recipes.pkl")
    
    # Save a sample for manual review
    high_confidence = df[df['label_confidence'] > 0.8].sample(n=min(100, len(df)), random_state=42)
    sample_columns = ['id', 'title', 'main_cat', 'calories_per_serving', 'health_label_ensemble', 'label_confidence']
    high_confidence[sample_columns].to_csv('data/processed/sample_for_review.csv', index=False)
    print("Sample for manual review saved to data/processed/sample_for_review.csv")
    
    return df


if __name__ == "__main__":
    df = main() 