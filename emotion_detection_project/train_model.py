import pandas as pd
import neattext as nt
import re
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv("data/emotions.csv")
print(data.head())
print(data.info())
print(data['emotion'].value_counts())

def clean_text(text):
    frame = nt.TextFrame(text)
    cleaned = frame.clean_text().lower()
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    return cleaned

data['text'] = data['text'].apply(clean_text)

X = data['text']
y = data['emotion']

vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

rf_best = grid_search.best_estimator_
rf_pred = rf_best.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandomForest Accuracy:", rf_accuracy)
print("\nClassification Report:\n", classification_report(y_test, rf_pred))

joblib.dump(rf_best, "emotions.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel and vectorizer saved successfully!")

def plot_confusion_matrix(y_true, y_pred, model_name, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

labels = rf_best.classes_
plot_confusion_matrix(y_test, rf_pred, "RandomForest", labels)
