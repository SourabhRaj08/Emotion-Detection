# Emotion Detection from Text
This project is a text-based emotion detection system that identifies emotions like **happy**, **sad**, and **angry** from given sentences. It uses **RandomForestClassifier** for accurate predictions and includes a cleaned and preprocessed dataset.

## Key Features
- Detects emotions from text input  
- Supports three emotions: `happy`, `sad`, and `angry`  
- Uses **RandomForestClassifier** for reliable predictions  
- Cleaned and preprocessed dataset for accuracy  
- Visualizes results with a confusion matrix  
- Pre-trained model saved using `joblib` for easy reuse  

## Technologies Used
- **Python:** Main programming language  
- **Pandas:** For handling and processing dataset  
- **Neattext:** Cleaning and preprocessing text  
- **Scikit-Learn:** Model training and evaluation  
- **Seaborn & Matplotlib:** Confusion matrix visualization  

## Dataset
The project uses a custom dataset (`emotions.csv`) containing sample sentences labeled with corresponding emotions:
- **Happy:** Positive and joyful sentences  
- **Sad:** Expressions of sorrow or disappointment  
- **Angry:** Sentences reflecting frustration or anger  

## How to Run the Project

### 1. Install Required Libraries  
Run the following command to install necessary libraries:
```bash
pip install -r requirements.txt
```

### 2.Train the Model
Run the following command:
```bash
python train_model.py
```

### 3.Run the Detection Demo
To test the model with custom text:
Run the following command:
```bash
python main.py
```
When prompted, enter the text you want to analyze.
The program will detect and display the predicted emotion.

## Folder Structure
- data/: Contains the emotions.csv dataset  
- train_model.py: Script to train and save the model  
- main.py: Script to test the model with custom text  
- emotions.pkl: Saved RandomForest model  
- vectorizer.pkl: Saved TF-IDF vectorizer  
