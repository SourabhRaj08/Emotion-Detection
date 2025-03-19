import joblib
import neattext as nt

emotion_model = joblib.load("emotions.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_emotion(text):
    cleaned_text = nt.TextFrame(text).clean_text()

    features = vectorizer.transform([cleaned_text])

    emotion = emotion_model.predict(features)[0]
    return emotion

def main():
    print("\n Text-Based Emotion Detection Demo \n")

    while True:
        text = input("Enter text (or 'q' to quit): ").strip()

        if text.lower() == 'q':
            print("\nExiting the demo. Goodbye! ")
            break

        emotion = predict_emotion(text)
        print(f"\nDetected Emotion: {emotion}\n")

if __name__ == "__main__":
    main()
