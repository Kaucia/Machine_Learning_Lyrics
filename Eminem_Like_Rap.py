#!/usr/bin/env python
# coding: utf-8

# In[46]:


import random
import numpy as np
import tensorflow as tf
#file_paths = ["C:\\Users\\Kaustubha\\Downloads\\ALL_eminem.txt"]
file_path = "C:\\Users\\Kaustubha\\Downloads\\ALL_eminem.txt"

try:
    with open(file_path, "r", encoding="utf-8") as file:
        text_content = file.read()
    print(text_content)
except UnicodeDecodeError:
    print("Unable to decode the file with 'utf-8' encoding. Please specify the correct encoding.")


# In[47]:



characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 5

sentences = []
next_char = []


# In[48]:


for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH,len(characters)), dtype=bool)
y = np.zeros((len(sentences),len(characters)), dtype=bool)

for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1


# In[49]:


import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

model = Sequential()
model.add(LSTM(128,input_shape=(SEQ_LENGTH,len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.01))

model.fit(x, y, batch_size=200, epochs=4)
model.save('EMrap')


# In[51]:




def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character

    # Post-processing to ensure generated words belong to the training text
        generated_words = generated.split()
        filtered_words = [word for word in generated_words if word in text]

    # If there are no words that match, generate a new word and check again
        if not filtered_words:
            continue  # Skip this iteration and try again

    # Join the filtered words to get the final generated text
        filtered_generated_text = ' '.join(filtered_words)

    return filtered_generated_text


temperatures = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]

for temp in temperatures:
    generated_text = generate_text(300, temp)
    print(f"Generated Text (Temperature {temp}):")
    print(generated_text)


# In[ ]:




