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

# Naive-Bayes models
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# for F1 scores/classification_reports
from sklearn.metrics import f1_score 

# -----------------------------------------------------------------------------------------------------------------------------------------------------
stop_words = set(stopwords.words('english'))
# Adding custom words to the stop-words list
cust_stop_words = ["'s"]
for temp in cust_stop_words:
    stop_words.add(temp)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Top-K Feature Selection
def feature_selection(x_traintf, y_train, x_testtf, k_best):
    selector = SelectKBest(mutual_info_classif, k=k_best)
    selector.fit(x_traintf, y_train)
    x_train = selector.transform(x_traintf)
    x_test = selector.transform(x_testtf)

    return x_train, x_test

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Naive-Bayes Model
def naive_bayes(x_traintf, y_train, x_testtf, y_test):
    k_best = [1,10,100,1000,10000]
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    scores = {}
    # Naive-Bayes Models
    temp_mnb, temp_bnb = [] , []
    print("Applying Models")
    for i in tqdm(range(len(k_best)), desc= "Progress"):
        entry = k_best[i]
        x_train, x_test = feature_selection(x_traintf, y_train, x_testtf, entry)
        bnb.fit(x_train.toarray(), y_train)
        mnb.fit(x_train.toarray(), y_train)
        yhat_mnb = mnb.predict(x_test.toarray())
        yhat_bnb = bnb.predict(x_test.toarray())
        f1_mnb = f1_score(y_test, yhat_mnb, average='macro')
        f1_bnb = f1_score(y_test, yhat_bnb, average='macro')
        temp_bnb.append(tuple((entry,f1_bnb)))
        temp_mnb.append(tuple((entry,f1_mnb)))
    scores["multi"] = temp_mnb
    scores["bernoulli"] = temp_bnb
    
    # # Bernoulli-NB
    # temp = []
    # print("Applying Bernoulli Naive-Bayes Model")
    # for i in tqdm(range(len(k_best)), desc= "Progress"):
    #     entry = k_best[i]
    #     x_train, x_test = feature_selection(x_traintf, y_train, x_testtf, entry)
    #     bnb.fit(x_train, y_train)
    #     yhat = bnb.predict(x_test)
    #     f1 = f1_score(y_test, yhat, average='macro')
    #     temp.append(tuple((entry,f1)))     
    # scores["bernoulli"] = temp

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

    scores = naive_bayes(train_mat, train_lbl, test_mat, test_lbl)

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
        temp = scores["multi"]
        
        # Multinomial-NB
        text = "MultinomialNB"
        heading = "NumFeature" + (" " * 6)
        for score in temp:
            heading = heading + str(score[0]) + (" " * 7)
            text = text + " | " + str(round(score[1],5))

        text_file.write(heading + "\n")
        text_file.write(text + "\n")

        # Bernoulli-NB
        temp = scores["bernoulli"]
        text = "BernoulliNB"
        for score in temp:
            text = text + " | " + str(round(score[1],5))
        
        text_file.write(text + "\n")

        text_file.write("\n")
# -----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()