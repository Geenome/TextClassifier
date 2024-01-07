import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set to '3' for no logs, '2' to show only warnings and errors

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.regularizers import l2

# Suppress TensorFlow deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Step 1: Read the file and split into training and testing sets
file_path = r'C:\Users\Mason\Documents\University Things\CE314\Assignment 2\IMDB Dataset.csv'
df = pd.read_csv(file_path)

# Print column names
print("Column Names:", df.columns)

# Assuming your CSV columns are named 'review' and 'sentiment'
comments = df['review'].values
sentiments = df['sentiment'].values

# Step 2: Pre-process the text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(comments)

sequences = tokenizer.texts_to_sequences(comments)
padded_sequences = pad_sequences(sequences, maxlen=200, truncating='post')

# Step 3: Build the bidirectional LSTM model with dropout
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    Bidirectional(LSTM(64, kernel_regularizer=l2(0.0001), recurrent_regularizer=l2(0.0001))),
    Dropout(0.5),  # Add dropout
    Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
    Dropout(0.5),  # Add dropout
    Dense(1, activation='sigmoid')
])

# Adjust learning rate and implement gradient clipping
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 4: Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, sentiments, test_size=0.2, stratify=sentiments, random_state=42)

history = model.fit(
    X_train,
    pd.Series(y_train).map({'positive': 1, 'negative': 0}),
    epochs=50,  # Increase the number of epochs
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Step 5: Evaluate the model
test_predictions = (model.predict(X_test) > 0.5).astype(int)
y_test_binary = pd.Series(y_test).map({'positive': 1, 'negative': 0})

print("Classification Report:\n", classification_report(y_test_binary, test_predictions))
print("Accuracy:", accuracy_score(y_test_binary, test_predictions))

# Visualize training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
