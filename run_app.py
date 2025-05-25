#!/usr/bin/env python3
"""
Simple launcher for the Turkish Recipe Health Classification System
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    
    print("🥗 Turkish Recipe Health Classification System")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        'streamlit_app.py',
        'models/best_recipe_classifier.pkl',
        'models/feature_engineering.pkl',
        'data/detailed_recipe_categorie_unitsize_calorie_chef.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        print("\nPlease ensure all required files are present.")
        return
    
    print("✅ All required files found")
    print("🚀 Starting Streamlit application...")
    print("\nThe app will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching application: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 