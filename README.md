# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

1.We aim to develop an LSTM-based neural network model using Bidirectional Recurrent Neural Networks for recognizing the named entities in the text.
2.The dataset used has a number of sentences, and each words have their tags.
3.We have to vectorize these words using Embedding techniques to train our model.
4.Bidirectional Recurrent Neural Networks connect two hidden layers of opposite directions to the same output.

## Dataset:
![image](https://github.com/Bmohamedathil/named-entity-recognition/assets/119560261/336c5ad4-9900-49c6-81ff-36c9809a6f5e)

## DESIGN STEPS

### STEP 1:
Import the necessary packages.

### STEP 2:
Read the dataset, and fill the null values using forward fill.

### STEP 3:
Create a list of words, and tags. Also find the number of unique words and tags in the dataset.

### STEP 4:
Create a dictionary for the words and their Index values. Do the same for the tags as well,Now we move to moulding the data for training and testing.

### STEP 5:
We do this by padding the sequences,This is done to acheive the same length of input data.

### STEP 6:
We build a build a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, Time Distributed Dense Layers

### STEP 7:
We compile the model and fit the train sets and validation sets,We plot the necessary graphs for analysis,A custom prediction is done to test the model manually.

## PROGRAM
```
Developed by: MOHAMED ATHIL B
Register No : 212222230081
```
### Libraries:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
from tensorflow.keras.preprocessing import sequence
```
### Read & Pre-Process Data:
```
data = pd.read_csv("ner_dataset.csv", encoding="latin1")

data.head(50)

data = data.fillna(method="ffill")

data.head(50)

print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())

words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())

print("Unique tags are:", tags)
```
### Define DClass to Get Sentence:
```
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = getter.sentences

len(sentences)

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

X1 = [[word2idx[w[0]] for w in s] for s in sentences]
```
### Padding:
```
X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)

y1 = [[tag2idx[w[2]] for w in s] for s in sentences]

y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)
```
### LSTM Model:
```
input_word = layers.Input(shape=(max_len,))

embedding_layer = layers.Embedding(input_dim=num_words,output_dim=50,
                                   input_length=max_len)(input_word)
dropout = layers.SpatialDropout1D(0.1)(embedding_layer)

bid_lstm = layers.Bidirectional(
    layers.LSTM(units=100,return_sequences=True,
                recurrent_dropout=0.1))(dropout)

output = layers.TimeDistributed(
    layers.Dense(num_tags,activation="softmax"))(bid_lstm)

model = Model(input_word, output)  

model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    x=X_train, y=y_train, validation_data=(X_test,y_test),
    batch_size=50, epochs=3,
)
```
### Metrics:
```
metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()
```
### Prediction:
```
i = 23
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![275592337-1dc998b6-daf6-41f4-8fb8-d244326bbc85](https://github.com/Bmohamedathil/named-entity-recognition/assets/119560261/84604bde-6926-402e-9513-a9b0f8ceb8df)

### Accuracy, Validation Accuracy Vs Iteration:
![275592445-d01d059c-b4ec-47ce-8b0f-0f984eeee30c](https://github.com/Bmohamedathil/named-entity-recognition/assets/119560261/487b5a0a-1863-47b1-9f52-03a6b48d151a)


### Sample Text Prediction
![275592587-1fcd0fac-dcf7-45d9-8cee-cae0d28c42b1](https://github.com/Bmohamedathil/named-entity-recognition/assets/119560261/97869c33-e37b-4939-bc6b-25e0597983c3)


![275592878-a766b904-baac-4851-b269-4df43278957d](https://github.com/Bmohamedathil/named-entity-recognition/assets/119560261/3f9fda7c-047e-48d5-867e-8ddc6d2f4a0d)

## RESULT:
Thus, an LSTM-based model for recognizing the named entities in the text is successfully developed.
