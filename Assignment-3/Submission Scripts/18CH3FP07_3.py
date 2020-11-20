'''
Arnesh Kumar Issar
18CH3FP07
'''

import os
import sys
import pickle5 as pickle
import nltk
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm


# Tokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
 
# Lemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Count-Vectorizer & TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
vectorizer = CountVectorizer()
tfidf = TfidfTransformer()

# KNN classifiers
from sklearn.neighbors import KNeighborsClassifier

# for F1 scores/classification_reports
from sklearn.metrics import f1_score 

# -----------------------------------------------------------------------------------------------------------------------------------------------------
stop_words = set(stopwords.words('english'))
# Adding custom words to the stop-words list
cust_stop_words = ["'s"]
for temp in cust_stop_words:
    stop_words.add(temp)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# KNN Classification
def knn(x_train, y_train, x_test, y_test):
    nn_list = [1,10,50]
    scores =[]

    print("Applying KNN Classifier")
    for i in tqdm(range(len(nn_list)), desc= "Progress"):
        entry = nn_list[i]
        knn = KNeighborsClassifier(n_neighbors=entry)
        knn.fit(x_train, y_train)
        yhat = knn.predict(x_test)
        f1 = f1_score(y_test, yhat, average='macro')
        scores.append(tuple((entry,f1)))

    return scores   


# ========================================================================================================================================
def main():
    data_path = sys.argv[1]
    
    # Forming train/test dataset for each class
    train_arr, train_lbl, test_arr, test_lbl, n_class1 = create_dataset(data_path)

    # Count-Vectorization
    vectorizer.fit(train_arr)
    train_mat = vectorizer.transform(train_arr)
    test_mat = vectorizer.transform(test_arr)

    # Tf-idf Transformer
    tfidf.fit(train_mat)
    train_tfmat = tfidf.transform(train_mat)
    test_tfmat = tfidf.transform(test_mat)

    scores = knn(train_tfmat, train_lbl, test_tfmat, test_lbl)

    write_scores(scores)

# ========================================================================================================================================
# Text cleanup
def cleanup_text(text):
    """
    Input: Un-processed text (str)
    Output: Processed text (str)
    """

    # Tokenizing
    tokens = tokenizer.tokenize(text.lower())
    filtered_tokens = []
    # Stop-Word Removal + Lemmatization
    for token in tokens:
        if(token not in stop_words):
            filtered_tokens.append(lemmatizer.lemmatize(token))
    
    return " ".join(filtered_tokens)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the dataset
def create_dataset(data_path):
    """
    Input: Dataset file-path
    Output: Tokenized Train/Test document string list + Train/Test Labels 
    """
    class1_path = data_path + "/class1"
    class2_path = data_path + "/class2"

    class1_train_file_list = os.listdir(class1_path + "/train")
    class1_test_file_list = os.listdir(class1_path + "/test")

    class2_train_file_list = os.listdir(class2_path + "/train")
    class2_test_file_list = os.listdir(class2_path + "/test")
    

    train_lbl= [1]*len(class1_train_file_list) + [2]*len(class2_train_file_list)
    test_lbl= [1]*len(class1_test_file_list) + [2]*len(class2_test_file_list)
    train_arr, test_arr = [], []

    print("Building Training & Testing Dataset")
    for i in tqdm(range(len(class1_train_file_list)), desc= "Class 1 Training Dataset"):
        file = class1_train_file_list[i]
        text = open(class1_path + "/train/" + file, 'rb').read().decode(errors='replace')
        train_arr.append(cleanup_text(text))
    
    for i in tqdm(range(len(class2_train_file_list)), desc= "Class 2 Training Dataset"):
        file = class2_train_file_list[i]
        text = open(class2_path + "/train/" + file, 'rb').read().decode(errors='replace')
        train_arr.append(cleanup_text(text))
    
    for i in tqdm(range(len(class1_test_file_list)), desc= "Class 1 Testing Dataset"):
        file = class1_test_file_list[i]
        text = open(class1_path + "/test/" + file, 'rb').read().decode(errors='replace')
        test_arr.append(cleanup_text(text))

    for i in tqdm(range(len(class2_test_file_list)), desc= "Class 2 Testing Dataset"):
        file = class2_test_file_list[i]
        text = open(class2_path + "/test/" + file, 'rb').read().decode(errors='replace')
        test_arr.append(cleanup_text(text))

    return train_arr, train_lbl, test_arr, test_lbl, len(class1_train_file_list) 

# -----------------------------------------------------------------------------------------------------------------------------------------------------
def write_scores(scores):
    with open(sys.argv[2],"a") as text_file:
        
        text = "KNN_Classifier"
        heading = "Num_Neighbours" + (" " * 3)
        for score in scores:
            heading = heading + str(score[0]) + (" " * 6)
            text = text + " | " + str(round(score[1],5))

        text_file.write(heading + "\n")
        text_file.write(text + "\n")

        text_file.write("\n")
# -----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()