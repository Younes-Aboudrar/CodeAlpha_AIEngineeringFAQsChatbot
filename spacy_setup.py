"""
SpaCy Setup Script for AI Engineering Chatbot
This script handles the installation and configuration of SpaCy and its language models.
"""

import sys
import subprocess
import json
import os

def check_spacy_installation():
    """Check if SpaCy is installed and import it if available."""
    try:
        import spacy
        return True
    except ImportError:
        print("SpaCy is not installed. Please install it using:")
        print("pip install spacy")
        print("\nAfter installation, run this script again to download a language model.")
        return False

def download_spacy_model():
    """Prompt user to select and download a SpaCy model."""
    print("\nSpaCy Model Selection")
    print("=====================")
    print("Please select a SpaCy model to download:")
    print("1. Small model (en_core_web_sm) - ~12MB - Fastest, least accurate")
    print("2. Medium model (en_core_web_md) - ~40MB - Balance of speed and accuracy")
    print("3. Large model (en_core_web_lg) - ~560MB - Slowest, most accurate")
    print("4. Fallback only - Skip SpaCy model download and use TF-IDF instead")
    
    choice = input("\nEnter your choice (1-4): ")
    
    models = {
        "1": "en_core_web_sm",
        "2": "en_core_web_md",
        "3": "en_core_web_lg",
        "4": "fallback"
    }
    
    if choice not in models:
        print("Invalid choice. Please run the script again and select a valid option (1-4).")
        return None
    
    selected_model = models[choice]
    
    # If fallback option is chosen, skip download
    if selected_model == "fallback":
        print("\nSkipping SpaCy model download. The application will use the TF-IDF fallback system.")
        save_model_config(selected_model)
        return selected_model
    
    print(f"\nDownloading {selected_model}...")
    
    try:
        # Run the SpaCy download command
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", selected_model],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "already installed" in result.stdout or "Download and installation successful" in result.stdout:
            print(f"\nSuccessfully installed {selected_model}!")
            save_model_config(selected_model)
            return selected_model
        else:
            print(f"\nUnexpected output during model installation: {result.stdout}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading model: {e}")
        print(f"Error details: {e.stderr}")
        print("\nPossible causes:")
        print("- Network connectivity issues")
        print("- Insufficient disk space")
        print("- Incompatible SpaCy version")
        print("- Missing build tools (for Windows users)")
        print("\nFor Windows users:")
        print("Some SpaCy models require Microsoft Visual C++ Build Tools.")
        print("You may need to install them from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        return None
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return None

def save_model_config(model_name):
    """Save the selected model name to a configuration file."""
    config_data = {"spacy_model": model_name}
    
    try:
        with open("config.json", "w") as config_file:
            json.dump(config_data, config_file, indent=4)
        print(f"Configuration saved to config.json")
    except Exception as e:
        print(f"Error saving configuration: {e}")

def main():
    """Main function to run the SpaCy setup process."""
    print("SpaCy Setup Utility")
    print("==================\n")
    
    if not check_spacy_installation():
        return
    
    # Verify SpaCy installation and import it
    try:
        import spacy
        print(f"SpaCy version {spacy.__version__} is installed.")
    except ImportError:
        # This should not happen as we already checked, but just in case
        print("Error importing SpaCy. Please reinstall it and try again.")
        return
    
    # Check if config.json already exists
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as config_file:
                config = json.load(config_file)
                if "spacy_model" in config:
                    print(f"\nA SpaCy model ({config['spacy_model']}) is already configured.")
                    choice = input("Do you want to download a different model? (y/n): ")
                    if choice.lower() != 'y':
                        print("\nKeeping existing model configuration.")
                        return
        except Exception as e:
            print(f"Error reading configuration file: {e}")
    
    # Download the SpaCy model or set fallback
    downloaded_model = download_spacy_model()
    
    if downloaded_model:
        print("\nSetup completed successfully!")
        if downloaded_model == "fallback":
            print("The application will use the TF-IDF fallback system for text processing.")
            print("Note: TF-IDF provides basic text matching without semantic understanding.")
        else:
            print(f"The application will use the {downloaded_model} SpaCy model.")
    else:
        print("\nSetup did not complete successfully.")
        print("Please try again or select option 4 for the fallback system.")

if __name__ == "__main__":
    main()
