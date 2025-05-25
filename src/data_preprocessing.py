import json
import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Any, Tuple
import nltk
from collections import Counter

# Turkish text preprocessing
import regex as re_turkish

class TurkishTextPreprocessor:
    """
    Turkish-specific text preprocessing for recipe classification
    """
    
    def __init__(self):
        self.setup_nltk()
        self.setup_turkish_stopwords()
        self.setup_ingredient_keywords()
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def setup_turkish_stopwords(self):
        """Turkish stopwords for recipe text"""
        self.turkish_stopwords = {
            'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz', 'on',
            've', 'veya', 'ile', 'için', 'kadar', 'gibi', 'daha', 'çok', 'az', 'biraz',
            'adet', 'gram', 'kilogram', 'litre', 'bardak', 'kaşık', 'çay', 'yemek', 'su',
            'tane', 'demet', 'dal', 'diş', 'tutam', 'paket', 'kutu', 'şişe',
            'dakika', 'saat', 'tarifi', 'tarif', 'nasıl', 'yapılır', 'malzemeler',
            'de', 'da', 'den', 'dan', 'te', 'ta', 'ten', 'tan', 'e', 'a', 'i', 'ı',
            'o', 'ö', 'u', 'ü', 'le', 'la', 'ler', 'lar', 'nin', 'nın', 'nun', 'nün'
        }
    
    def setup_ingredient_keywords(self):
        """Define ingredient categories for feature engineering"""
        self.healthy_keywords = {
            'sebze': ['domates', 'salatalık', 'marul', 'soğan', 'sarımsak', 'havuç', 'kabak', 'patlıcan', 'biber', 'brokoli', 'ıspanak', 'roka', 'maydanoz', 'dereotu', 'nane'],
            'meyve': ['elma', 'armut', 'portakal', 'limon', 'üzüm', 'çilek', 'muz', 'kivi', 'ananas', 'şeftali'],
            'protein': ['tavuk', 'balık', 'hindi', 'yumurta', 'baklagil', 'mercimek', 'nohut', 'fasulye'],
            'tahıl': ['bulgur', 'kinoa', 'yulaf', 'esmer pirinç', 'tam buğday'],
            'süt_ürünü': ['yoğurt', 'süt', 'kefir', 'cottage', 'lor']
        }
        
        self.unhealthy_keywords = {
            'işlenmiş': ['sosis', 'salam', 'jambon', 'sucuk', 'pastırma', 'hazır', 'konserve'],
            'kızartma': ['kızartma', 'fritür', 'bol yağ', 'derin yağ'],
            'şeker': ['şeker', 'şurup', 'bal', 'reçel', 'çikolata', 'krema'],
            'beyaz_un': ['beyaz un', 'makarna', 'ekmek', 'börek', 'hamur'],
            'yağ': ['tereyağı', 'margarin', 'kuyruk yağı', 'ayçiçek yağı']
        }
    
    def normalize_turkish_text(self, text: str) -> str:
        """Normalize Turkish text"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers that are standalone
        text = re.sub(r'\b\d+\b', '', text)
        
        return text.strip()
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Remove Turkish stopwords"""
        return [word for word in words if word not in self.turkish_stopwords]
    
    def tokenize_turkish(self, text: str) -> List[str]:
        """Tokenize Turkish text"""
        text = self.normalize_turkish_text(text)
        words = text.split()
        words = self.remove_stopwords(words)
        return [word for word in words if len(word) > 2]
    
    def extract_ingredient_features(self, ingredients: List[str]) -> Dict[str, int]:
        """Extract features from ingredients"""
        features = {}
        all_ingredients_text = ' '.join(ingredients).lower()
        
        # Count healthy ingredient categories
        for category, keywords in self.healthy_keywords.items():
            count = sum(1 for keyword in keywords if keyword in all_ingredients_text)
            features[f'healthy_{category}_count'] = count
        
        # Count unhealthy ingredient categories
        for category, keywords in self.unhealthy_keywords.items():
            count = sum(1 for keyword in keywords if keyword in all_ingredients_text)
            features[f'unhealthy_{category}_count'] = count
        
        return features


class RecipeDataLoader:
    """
    Load and preprocess Turkish recipe data
    """
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.preprocessor = TurkishTextPreprocessor()
        
    def load_data(self) -> pd.DataFrame:
        """Load JSON data into pandas DataFrame"""
        print("Loading recipe data...")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} recipes")
        
        return df
    
    def extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numeric features from text fields"""
        
        # Extract calories with smart defaults for missing values
        df['calories_numeric'] = df.apply(self._extract_calories_with_defaults, axis=1)
        
        # Extract times
        df['prep_time_minutes'] = df['preparing_time'].apply(self._extract_time)
        df['cook_time_minutes'] = df['cooking_time'].apply(self._extract_time)
        df['total_time_minutes'] = df['prep_time_minutes'] + df['cook_time_minutes']
        
        # Extract serving size
        df['serving_size'] = df['size'].apply(self._extract_serving_size)
        
        # Calories per serving (already corrected, no need to divide again)
        df['calories_per_serving'] = df['calories_numeric']
        
        return df
    
    def _extract_calories(self, calorie_str: str) -> float:
        """Extract numeric calories from string"""
        if pd.isna(calorie_str) or not isinstance(calorie_str, str) or calorie_str.lower() == 'nan':
            return 0
        
        # Find numbers in calorie string
        numbers = re.findall(r'\d+', calorie_str)
        return float(numbers[0]) if numbers else 0
    
    def _extract_calories_with_defaults(self, row) -> float:
        """Extract calories with category-based defaults for missing values"""
        calorie_str = row.get('calorie', '0')
        calories = self._extract_calories(calorie_str)
        
        # Apply smart defaults for missing/zero calories
        if pd.isna(calories) or calories == 0:
            category = str(row.get('main_cat', '')).lower()
            if 'tatli' in category or 'kek' in category or 'kurabiye' in category:
                calories = 350  # Default for desserts
            elif 'salata' in category or 'meze' in category or 'sebze' in category:
                calories = 150  # Default for salads/appetizers/vegetables
            elif 'çorba' in category:
                calories = 200  # Default for soups
            elif 'et' in category or 'tavuk' in category:
                calories = 300  # Default for meat dishes
            else:
                calories = 250  # General default
        
        return calories
    
    def _extract_time(self, time_str: str) -> float:
        """Extract time in minutes"""
        if pd.isna(time_str) or not isinstance(time_str, str):
            return 0
        
        # Extract hours and minutes
        hours = re.findall(r'(\d+)\s*saat', time_str)
        minutes = re.findall(r'(\d+)\s*dakika', time_str)
        
        total_minutes = 0
        if hours:
            total_minutes += int(hours[0]) * 60
        if minutes:
            total_minutes += int(minutes[0])
            
        return total_minutes
    
    def _extract_serving_size(self, size_str: str) -> int:
        """Extract serving size number"""
        if pd.isna(size_str) or not isinstance(size_str, str):
            return 4  # default
        
        numbers = re.findall(r'\d+', size_str)
        return int(numbers[0]) if numbers else 4
    
    def preprocess_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess text features"""
        
        # Process title
        df['title_tokens'] = df['title'].apply(
            lambda x: self.preprocessor.tokenize_turkish(x) if pd.notna(x) else []
        )
        
        # Process ingredients
        df['ingredients_text'] = df['ingredients'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )
        df['ingredients_tokens'] = df['ingredients_text'].apply(
            self.preprocessor.tokenize_turkish
        )
        
        # Extract ingredient features
        ingredient_features = df['ingredients'].apply(
            self.preprocessor.extract_ingredient_features
        )
        ingredient_features_df = pd.DataFrame(ingredient_features.tolist()).fillna(0)
        
        # Combine with main dataframe
        df = pd.concat([df, ingredient_features_df], axis=1)
        
        return df
    
    def create_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for health classification"""
        
        # Ingredient counts
        df['ingredient_count'] = df['ingredients'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # Healthy vs unhealthy ingredient ratio
        healthy_cols = [col for col in df.columns if col.startswith('healthy_')]
        unhealthy_cols = [col for col in df.columns if col.startswith('unhealthy_')]
        
        df['healthy_ingredient_total'] = df[healthy_cols].sum(axis=1)
        df['unhealthy_ingredient_total'] = df[unhealthy_cols].sum(axis=1)
        
        # Health ratio
        df['health_ratio'] = df['healthy_ingredient_total'] / (
            df['healthy_ingredient_total'] + df['unhealthy_ingredient_total'] + 1
        )
        
        # Cooking method indicators
        df['is_fried'] = df['ingredients_text'].str.contains(
            'kızart|fritür|derin yağ', na=False
        ).astype(int)
        
        df['is_baked'] = df['ingredients_text'].str.contains(
            'fırın|pişir', na=False
        ).astype(int)
        
        df['is_raw'] = df['ingredients_text'].str.contains(
            'çiğ|salata|mezze', na=False
        ).astype(int)
        
        return df
    
    def process_data(self) -> pd.DataFrame:
        """Complete data processing pipeline"""
        # Load data
        df = self.load_data()
        
        # Extract numeric features
        df = self.extract_numeric_features(df)
        
        # Preprocess text features
        df = self.preprocess_text_features(df)
        
        # Create health-specific features
        df = self.create_health_features(df)
        
        # Save processed data
        df.to_pickle('data/processed/processed_recipes.pkl')
        print("Processed data saved to data/processed/processed_recipes.pkl")
        
        return df


def main():
    """Main preprocessing pipeline"""
    # Load and process data
    loader = RecipeDataLoader('data/detailed_recipe_categorie_unitsize_calorie_chef.json')
    df = loader.process_data()
    
    # Display basic statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total recipes: {len(df)}")
    print(f"Average calories: {df['calories_numeric'].mean():.1f}")
    print(f"Average prep time: {df['prep_time_minutes'].mean():.1f} minutes")
    print(f"Average cook time: {df['cook_time_minutes'].mean():.1f} minutes")
    
    print("\n=== Main Categories ===")
    print(df['main_cat'].value_counts().head(10))
    
    print("\n=== Health Features Summary ===")
    print(f"Healthy ingredient avg: {df['healthy_ingredient_total'].mean():.2f}")
    print(f"Unhealthy ingredient avg: {df['unhealthy_ingredient_total'].mean():.2f}")
    print(f"Health ratio avg: {df['health_ratio'].mean():.2f}")
    
    return df


if __name__ == "__main__":
    df = main() 