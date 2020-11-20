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

# for F1 scores/classification_reports
from sklearn.metrics import f1_score 

# -----------------------------------------------------------------------------------------------------------------------------------------------------
stop_words = set(stopwords.words('english'))
# Adding custom words to the stop-words list
cust_stop_words = ["'s"]
for temp in cust_stop_words:
    stop_words.add(temp)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Rocchio Algorithm
def rocchio(x_train, y_train, x_test, y_test, n_class1):
    b_list = [0.00]
    centroid = np.zeros((2,x_train.shape[1]))
    centroid[0] = np.sum(x_train[:n_class1], axis=0)/n_class1
    centroid[1] = np.sum(x_train[n_class1:], axis=0)/(len(y_train)-n_class1)
    
    scores = []
    print("Applying Rocchio Classifier")
    for i in tqdm(range(len(b_list)), desc= "Progress"):
        entry = b_list[i]
        yhat = np.zeros((x_test.shape[0],1))
        for i in range(x_test.shape[0]):
            dist_1 = np.linalg.norm(centroid[0] - x_test[i])
            dist_2 = np.linalg.norm(centroid[1] - x_test[i])
            if(dist_1 < (dist_2 - entry)):
                yhat[i] = 1
            
            # Doubt =================================================================
            # elif(dist_2 < (dist_1 - entry)):
            #     yhat[i] = 2
            # elif(dist_1 < dist_2):
            #     yhat[i] = 1
            # Doubt =================================================================

            else:
                yhat[i] = 2
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

    scores = rocchio(train_tfmat.toarray(), train_lbl, test_tfmat.toarray(), test_lbl, n_class1)

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
        
        text = "Rocchio"
        heading = "b" + (" " * 9)
        for score in scores:
            heading = heading + str(score[0]) + (" " * 6)
            text = text + " | " + str(round(score[1],5))

        text_file.write(heading + "\n")
        text_file.write(text + "\n")

        text_file.write("\n")
# -----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()