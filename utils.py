import contractions
import gensim.downloader as api
import math
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
import math
import fasttext
import fasttext.util
import gensim

tk = RegexpTokenizer(r'\w+')
schemas = ["vulnerable", "angry", "impulsive", "happy", "detached", "punishing", "healthy"]
max_epochs = 30
vec_size = 500


def tokenizer(texts: list) -> list:
    tokenized_texts = []
    for i in range(len(texts)):
        words = tk.tokenize(texts[i])
        tokenized_texts.append(words)
    return tokenized_texts


def remove_stopwords(s: str) -> str:
    new_str = ""
    for word in s.split():
        if word not in stopwords.words('english'):
            new_str += word + " "
    return new_str


def split_data(input_x, label_y, percent: float) -> (list, list, list, list):
    # Before
    # num_of_train = math.floor(len(data) * percent)
    # train_data = []
    # test_data = []
    # for i in range(0, num_of_train):
    #     train_data.append(data[i])
    #
    # for i in range(num_of_train, len(data)):
    #     test_data.append(data[i])

    # After
    x_train, y_train, x_test, y_test = iterative_train_test_split(input_x, label_y, test_size=percent)
    return x_train, y_train, x_test, y_test, percent


# Return list of tokenized strings through pre-processing(lowercase, noise removal, stop-word removal)
def pre_process_data(texts: list) -> (list, list):
    # Convert all to lowercase
    processed_texts = list(map(lambda s: s.lower(), texts))

    # Noise removal
    processed_texts = list(map(lambda s: contractions.fix(s), processed_texts))

    # TODO: Spelling correction
    # TODO: remove numbers
    # TODO: remove punctuation

    # Stop word-removal
    processed_texts = list(map(lambda s: remove_stopwords(s), processed_texts))

    # Tokenizer of strings
    tokenized_texts = tokenizer(processed_texts)

    return processed_texts, tokenized_texts


# takes in dataframe, returns list of 'Texts' and list of 'Labels'
def get_text_labels(dataframe):
    rows, cols = (dataframe.shape[0], dataframe.shape[1])

    text_list = []
    label_list = np.zeros((rows, len(schemas)))

    texts = dataframe['Text']
    for txt in texts:
        text_list.append(txt)

    is_vulnerable = dataframe['is_vulnerable']
    is_angry = dataframe['is_angry']
    is_impulsive = dataframe['is_impulsive']
    is_happy = dataframe['is_happy']
    is_detached = dataframe['is_detached']
    is_punishing = dataframe['is_punishing']
    is_healthy = dataframe['is_healthy']

    for i in range(dataframe.shape[0]):
        j = 0
        label_list[i][j] = 1 if bool(is_vulnerable[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_angry[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_impulsive[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_happy[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_detached[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_punishing[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_healthy[i]) == True else 0

    return text_list, label_list


def get_average_for_each_label(dataframe):
    rows, cols = (dataframe.shape[0], dataframe.shape[1])
    text_list = []

    texts = dataframe['Text']
    for txt in texts:
        text_list.append(txt)

    average_label_list = np.zeros((rows, len(schemas)))
    for i in range(dataframe.shape[0]):
        j = 0
        average_label_list[i][j] = avg_helper(dataframe,i, 5, 15)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 16, 26)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 27, 35)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 36, 46)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 47, 56)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 57, 67)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 68, 78)

    return text_list, average_label_list


def avg_helper(dataframe, i, begin, end):
    mean = dataframe.iloc[i, begin:end].mean()
    for j in dataframe.iloc[i, begin:end]:
        if (j is 5 or j is 6) and mean < 3.5:
            mean = 3.5
    return get_label(mean)


def get_label(mean) -> int:
    mean = round(mean)
    if mean <= 3:
        return 0
    elif 3 < mean <= 4:
        return 1
    elif 4 < mean <= 5:
        return 2
    elif 5 < mean <= 6:
        return 3
    else:
        return 0

# TODO: train models
def training_model_fast_text():
    # If model is obtained, no need to run this part of code\
    # fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')
    return ft

# train a model on
def training_model_d2v(tokenized_texts):
    print("TAGGING DOCUMENTS")
    tagged_docs = []
    # associate tag to each document
    for i, sentence in enumerate(tokenized_texts):
        tagged_docs.append(gensim.models.doc2vec.TaggedDocument(sentence, [i]))
        i += 1

    print("TRAINING MODEL")
    model = gensim.models.Doc2Vec(documents=tagged_docs, vector_size=vec_size, window=10, epochs=max_epochs,
                                  min_count=1,
                                  workers=4, alpha=0.025, min_alpha=0.025)
    model.save('../models/schema-d2v-knn.model')

# returns pre-trained word2vec model
def get_word2vec():
    print('LOAD GOOGLE WORD VECTORS')

    return api.load("word2vec-google-news-300")

