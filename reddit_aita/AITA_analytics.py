import psycopg2
from psycopg2 import Error
from collections import defaultdict
sys.path.append('\\.')
from user_variables import password
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from sklearn.utils import compute_class_weight
from keras.optimizers import SGD, RMSprop

def extract_from_db():
    """
    Extracts data from db and return pandas DataFrame
    """

    try:
        connection = psycopg2.connect(user = "postgres",
                                    password = password,
                                    host = "127.0.0.1",
                                    port = "5432",
                                    database = "postgres")

        cursor = connection.cursor()
        cursor.execute("""SELECT date, text, tag, title FROM reddit.submissions
                        WHERE sub_reddit = 'AmItheAsshole' AND
                        tag IN ('not the a-hole', 'everyone sucks', 'no a-holes here', 'asshole');""")

        data_output = cursor.fetchall()

        data = pd.DataFrame(
            data_output,
            columns = ['date','text','flair','title']
        )

        return data

    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)

    finally:
        if(connection):
            cursor.close()
            connection.close()

def find_most_common_words(text_list: list, n_common_words):
    """
    Returns the n most common words in a list of str type data
    """

    word_dict = defaultdict(int)

    for text in text_list:
        for word in text.strip().split(" "): #Use regex to .replace() special chars
            word_dict[word.lower()] += 1

    words = [(key, value) for key, value in word_dict.items()]
    sorted_words = sorted(words, key = lambda w: w[1], reverse = True)[:n_common_words]

    most_common_words = [el[0] for el in sorted_words]

    return most_common_words

def create_word_dict(words: list):
    """
    Takes a list of unique words and returns a dictionary of words with a 
    unique int assigned to each word
    """

    if(len(words) != len(set(words))):
        raise ValueError("'words' must be a list of unique strings")

    encoded_dict = {}
    
    for i, word in enumerate(words):
        encoded_dict[word] = i

    return encoded_dict

def encode_text(text_list, n_common_words):
    """
    Encodes a list of str type data into a list of ints and returns list of int lists.
    """

    common_words = find_most_common_words(text_list, n_common_words)
    word_dict = create_word_dict(common_words)

    encoded_list = []

    for text in text_list:
        encoded_text = []

        for word in text.strip().split(" "):
            code = word_dict.get(word)

            if code:
                encoded_text.append(code)
                
        encoded_list.append(encoded_text)
            
    return encoded_list

def encode_labels(labels):
    """
    Encodes a list of str type data into a list of ints and returns int list.
    """

    encoded_labels = []

    word_dict = create_word_dict(list(set(labels))) #unique values

    for label in labels:
        code = word_dict.get(label.strip())
        encoded_labels.append(code)

    return encoded_labels, word_dict

def vectorize_sequence(data, length):
    """
    Return numpy array of one hot encoded data
    """

    results = np.zeros((len(data), length))

    for i, code in enumerate(data):
        results[i, code] = 1

    return results
    
def plot_accuracy(model):
    loss = model.history['loss']
    epochs = range(1, len(loss) + 1)

    acc = model.history['accuracy']
    val_acc = model.history['val_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

##Run pre-process
data = extract_from_db()
data['full_text'] = data['title'] + " " + data['text']

#Here we are adjusting the flair to predict if there is an ahole at all or not
data['adj_flair'] = data.flair.replace({'everyone sucks':'ahole', 
                                        'no a-holes here': 'not the a-hole', 
                                        'asshole':'ahole'})

#Even out classes by removing data (to test)
#indexes = data.index[data.flair == 'not the a-hole'].tolist()
#class_counts = data.flair.value_counts()
#last_index = min(round(class_counts[1] * 1.5), class_counts[0])
#indexes = indexes[int(last_index):]
#data = data.drop(index = indexes)

#input shape
input_shape = 3000
max_len = 1000

#Use tokenizer
text_tokenizer = Tokenizer(num_words=input_shape)
text_tokenizer.fit_on_texts(data.full_text)
text_sequences = text_tokenizer.texts_to_sequences(data.full_text)
text_sequences = sequence.pad_sequences(text_sequences, maxlen = max_len, padding = 'post')
text_one_hot = text_tokenizer.texts_to_matrix(data.full_text, mode='binary')

#Labels
labels, label_map = encode_labels(data.adj_flair)
one_hot_labels = vectorize_sequence(labels, 2)

#Split validation
text_train_set, text_test_set, label_train_set, label_test_set = train_test_split(text_one_hot, one_hot_labels, test_size = 0.2)

#Counts of each label
counts = np.apply_along_axis(sum, arr = one_hot_labels, axis = 0)

#Weights
weights = 1/(counts/sum(counts))

#Optimzer
opt = RMSprop(learning_rate=0.0001)

##########Dense Model##########
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(3000,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))
model.compile(optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy'])

history = model.fit(text_train_set,
                    label_train_set,
                    epochs=100,
                    batch_size = 64,
                    validation_data=(text_test_set, label_test_set))

plot_accuracy(history)

for i, prediction in enumerate(model.predict(text_test_set)):
    print(prediction, label_test_set[i])

#Evaluate model
score = model.evaluate(text_test_set, label_test_set, verbose=1)
model.predict(text_test_set)
print(score)

########LSTM model#########

#weights = compute_class_weight('balanced', list(label_map.keys()), data.flair)

text_train_set, text_test_set, label_train_set, label_test_set = train_test_split(text_sequences, one_hot_labels, test_size = 0.2)

model = models.Sequential()
model.add(layers.Embedding(input_shape, 128, input_length = max_len))
model.add(layers.LSTM(128))
model.add(layers.Dense(2, activation='sigmoid'))

model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

lstm_model = model.fit(text_train_set,
                    label_train_set,
                    epochs=20,
                    batch_size = 128,
                    validation_data=(text_test_set, label_test_set))

plot_accuracy(lstm_model)

lstm_score = model.evaluate(text_test_set, label_test_set, batch_size = 32)