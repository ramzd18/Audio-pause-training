import speech_recognition as sr
from tensorflow.keras.models import load_model
import pickle

# Load the model
loaded_model = load_model('next_token_model1.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
    
def predict_next_in_sequence(model, text, tokenizer, max_len):
    # Tokenize the input text
    sequence = tokenizer.texts_to_sequences([text])
    print(sequence)
#     padded_sequence = pad_sequences(sequence, maxlen=max_len)

    # Make prediction
    predictions = model.predict(sequence)[0]

    # Assuming classes: 0 - 'O', 1 - 'P', 2 - 'E'
    class_labels = {0: 'O', 1: 'P', 2: 'E'}

    # Process predictions
#     predicted_classes = np.argmax(predictions, axis=1)
#     labeled_predictions = [class_labels[cls] for cls in predicted_classes]

    return predictions

# # Example usage
# text="Can you find out when the next station uhm uhm" 
# predicted_classes = predict_next_in_sequence(loaded_model, text, loaded_tokenizer, 11)
# print(predicted_classes)

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Create a single microphone instance
microphone = sr.Microphone()

def recognize_speech_from_microphone(recognizer, microphone):
    with microphone as source:
        audio = recognizer.listen(source)
        try:
            # Recognize speech using Google Speech Recognition
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            # Google Speech Recognition could not understand audio
            return None
        except sr.RequestError:
            # Could not request results from Google Speech Recognition service
            return None

# Main loop for live speech recognition
while True:
    # Convert speech to text
    text = recognize_speech_from_microphone(recognizer, microphone)
    if text is not None:
        print("You said: " + text)
        predicted_classes = predict_next_in_sequence(loaded_model, text, loaded_tokenizer, 11)
        pause= predicted_classes[0]
        end_speech= predicted_classes[1]
        if pause >.35: 
            print("Predicted Pause")
        else: 
            print("End Sentence")

        
