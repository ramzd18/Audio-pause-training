import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
import sentencepiece as spm



def remove_tokens(text, tokens=["<pause>", "</s>"]):
    for token in tokens:
        text = text.replace(token, "")
    return text

file_path = "processed_refined_conversational_data (2).txt"
with open(file_path, 'r') as file:
    data = file.readlines()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)  # Fit on original data

X = []
Y = []

for sentence in data:
    words = sentence.split()
    sequence = []
    for word in words:
        tokenized_word = tokenizer.texts_to_sequences([word])[0]
        if tokenized_word:  # Check if the word was tokenized successfully
            sequence.append(tokenized_word[0])
            label = 'O'  # Default label
            if '<pause>' in word:
                label = 'P'
            elif '</s>' in word:
                label = 'E'
        if label == 'P' or label == 'E':
            X.append(sequence)
            if label == 'P':
                Y.append(0)  # Assign 1 to 'P' (pause)
            elif label == 'E':
                Y.append(1)  # Assign 2 to 'E' (end of speech)


spm.SentencePieceTrainer.train(input="processed_refined_conversational_data (2).txt", model_prefix='m', vocab_size=6000)
sp = spm.SentencePieceProcessor()
sp.load('m.model')
num_wordpieces = sp.get_piece_size()


X = np.array([np.array(xi) for xi in X])
print(X.shape)
Y = np.array(Y)
Y = to_categorical(Y, num_classes=2)
print(Y.shape)
# Split Data
train_sequences, val_sequences, train_labels, val_labels = train_test_split(X, Y, test_size=0.2, random_state=42)
print(len(train_sequences))
print(len(train_labels))
# untokenized_sentence = tokenizer.sequences_to_texts([tokenized_sequence])[0]

# print(tokenizer.sequences_to_texts(train_sequences[0]))
print(train_sequences[0])
print(train_labels[0])
num_labels = 2  # Adding 2 for <pause> and </s>

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)))

# Additional fully connected network as an additional predictor
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Adjusted output layer
model.add(Dense(num_labels, activation='softmax', kernel_regularizer=l2(0.01)))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model = Sequential()
# model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128))
# model.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
# model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)))
# model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.01)))

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
class_weights = {0: 1.2, 1: .55}


def data_generator(sequences, labels,batch_size=32):
    num_samples = len(sequences)
    while True:  
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            batch_sequences = pad_sequences(sequences[batch_indices], padding='post')
            batch_labels = labels[batch_indices]
            yield tf.convert_to_tensor(batch_sequences, dtype=tf.int32), tf.convert_to_tensor(batch_labels, dtype=tf.float32)

model.fit(data_generator(train_sequences, train_labels, batch_size=32),
          steps_per_epoch=len(train_sequences) // 32,
          epochs=3,  # Increased epochs
          validation_data=data_generator(val_sequences, val_labels, batch_size=32),
          validation_steps=len(val_sequences) // 32,
          class_weight=class_weights,
          callbacks=[early_stopping])

print("Before Saving")
model.save('next_token_model1.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
