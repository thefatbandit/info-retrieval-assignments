'''
Arnesh Kumar Issar
18CH3FP07
'''

import os
import math
import json
import sys
from collections import Counter
import pickle5 as pickle
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Tokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
 
# Lemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

file_path = os.getcwd()
parent_dir = os.path.dirname(os.getcwd())
data_path = parent_dir + "/Dataset/Dataset"

file_list = os.listdir(data_path)

stop_words = set(stopwords.words('english'))
# Adding custom words to the stop-words list
cust_stop_words = ["'s"]
for temp in cust_stop_words:
    stop_words.add(temp)

with open(parent_dir + '/Dataset/Leaders.pkl','rb') as fp:
    global leaders
    leaders = pickle.load(fp)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Converting the scored-documents to text lines for the result.txt
def list_to_string(score_list):
    text = ""
    if(len(score_list)==0):
        text = "No matching Documents found" + "\n" 
    else:
        for doc in score_list:
            if(round(doc[1],5)==0):
                text = text + '<' + str(doc[0]) + ',' + str(doc[1]) + '>,'
            else:
                text = text + '<' + str(doc[0]) + ',' + str(round(doc[1],5)) + '>,'

        text = text[:-1] + '\n'
    return text

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Finding Local & Global Champion Lists
def calc_champ():
    # Champion List Local
    champ_list_local = {}
    for token in inv_pos_index:
        temp = inv_pos_index[token][1:]
        temp.sort(key = lambda x: x[1],reverse = True)
        temp = [tuple(doc) for doc in temp[:min(50,len(temp))]]
        champ_list_local[token] = temp


    # Champion List Global
    champ_list_global = {}
    with open(parent_dir + '/Dataset/StaticQualityScore.pkl','rb') as fp:
        g = pickle.load(fp)

        for token in inv_pos_index:
            temp = inv_pos_index[token][1:]
            temp.sort(key = lambda x: x[1] + g[x[0]],reverse = True)
            temp = [tuple(doc)  for doc in temp[:min(50,len(temp))]]        
            champ_list_global[token] = temp
    
    return champ_list_global, champ_list_local

# -----------------------------------------------------------------------------------------------------------------------------------------------------
def calc_Vd():
    # Calculating sum of squares
    Vd_norm = [0]*len(file_list)

    for token in inv_pos_index:
        temp = inv_pos_index[token]
        for i in range(1,len(temp)):
            doc = temp[i][0]
            tidf = temp[i][1] * temp[0]
            Vd_norm[doc] = Vd_norm[doc] + (tidf**2)
        
    #  Calculating sqrt for Vd_norm
    for i in range(len(Vd_norm)):
        Vd_norm[i] = Vd_norm[i] ** 0.5
    
    return Vd_norm

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Function for finding followers to each leader
def find_followers(leaders, word_vec, lead_inv_pos_index):
    follow_dict = {}
    for i in tqdm(range(len(file_list)), desc= "Finding Leaders"):
        leader_score = []
        if(i in leaders): 
            pass
        else:
            leader = tf_idf(word_vec[i],lead_inv_pos_index)[0]
            if(leader[0] not in follow_dict):
                follow_dict[leader[0]] = [i]
            else:
                follow_dict[leader[0]].append(i)
    
    return follow_dict

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Main Ranked Retrieval Function 
def ranked_retrieval(champ_list_global, champ_list_local, follow_list):
    with open(sys.argv[1],"r") as query_file:  # For taking the commandline input for the query text file
        query_list = query_file.readlines()
    
    with open("RESULTS2_18CH3FP07.txt","wt") as text_file:
        for i in tqdm(range(len(query_list)), desc= "Ranked Retrieval"):
            query = query_list[i]
            filtered_query = []
            tf_idf_score = []
            
   
            query = query.rstrip("\n")

            # For blank lines
            if(query==""):
                text_file.write("Empty Query \n \n")
                print()
                continue
            
            else:
                print("Retrieval for " + '"' + query + '"' + " being done" ) 

                # Tokenizing
                tokens = tokenizer.tokenize(query.lower())

                # Stop-Word Removal + Lemmatization
                for token in tokens:
                    if(token not in stop_words):
                        filtered_query.append(lemmatizer.lemmatize(token))

                text_file.write(query + "\n")
                if(len(filtered_query)==0):
                    text_file.write("No matching Documents found \n \n")
                else:
                    tf_idf_score = tf_idf(filtered_query,inv_pos_index)
                    champ_local_score = champ(filtered_query, champ_list_local)
                    champ_global_score = champ(filtered_query, champ_list_global)
                    cluster_score = cluster_scoring(filtered_query, follow_list, lead_inv_pos_index)
                    #  Writing query data
                    text_file.write(list_to_string(tf_idf_score))
                    text_file.write(list_to_string(champ_local_score))
                    text_file.write(list_to_string(champ_global_score))  
                    text_file.write(list_to_string(cluster_score))  

                    text_file.write("\n")   

# ========================================================================================================================================
def main():
    global inv_pos_index
    inv_pos_index = {}
    global lead_inv_pos_index
    lead_inv_pos_index = {}
    word_vec = []
    
    # Buillding the & storing the inv_pos_index
    for i in tqdm(range(len(file_list)), desc="Building Positional Index"):
        filtered_tokens = []
        txt_file = open(data_path + "/" + str(i) + ".html")
        soup = BeautifulSoup(txt_file, features="html.parser")
        text = soup.get_text()

        # Tokenizing
        tokens = tokenizer.tokenize(text.lower())

        # Stop-Word Removal + Lemmatization
        for token in tokens:
            if(token not in stop_words):
                filtered_tokens.append(lemmatizer.lemmatize(token))
        
        # Adding the doc_text for finding leaders
        word_vec.append(filtered_tokens)
        
        # Finding term-freq of each token in the token-list
        token_freq = Counter(filtered_tokens)

        # Adding the (d,tf) pair to the Inverted Positional Index
        for token in token_freq:
            tf = math.log10(1+token_freq[token])
            if(token not in inv_pos_index):
                inv_pos_index[token] = [0,tuple((i,round(tf,5)))]
                # Adding to the leader inv_pos_index
                if(i in leaders):
                    lead_inv_pos_index[token] = [0,tuple((i,round(tf,5)))]
            else:
                inv_pos_index[token].append(tuple((i,round(tf,5))))
                # Adding to the leader inv_pos_index            
                if(i in leaders and token not in lead_inv_pos_index):
                    lead_inv_pos_index[token] = [0,tuple((i,round(tf,5)))]
                elif(i in leaders and token in lead_inv_pos_index):
                    lead_inv_pos_index[token].append(tuple((i,round(tf,5))))

    # Computing & Storing the idf value for each term
    for token in inv_pos_index:
        df = len(inv_pos_index[token])-1
        idf = math.log10(float(len(file_list)/df))
        inv_pos_index[token][0] = idf

    #  Setting idf values for leader inv_pos_index
    for token in lead_inv_pos_index:
        lead_inv_pos_index[token][0] = inv_pos_index[token][0]

    # Storing the inv_pos_index
    print("Storing the Index")
    with open("Inv_Pos_Index.json", 'w') as fp:
        json.dump(inv_pos_index, fp, sort_keys=True, indent=3)

    # Calculating global/local champ list
    champ_list_global, champ_list_local = calc_champ()

    global Vd_norm
    Vd_norm= calc_Vd()

    # Finding followers of each leader
    print("Building the Leader-Follower list")
    follow_list = find_followers(leaders, word_vec, lead_inv_pos_index)

    with open("Leaders.json", 'w') as fp:
        json.dump(follow_list, fp, sort_keys=True, indent=3)

    # Main Retreival Function
    print("Ranked Retrival Initiated")
    ranked_retrieval(champ_list_global, champ_list_local, follow_list)
    print("All documents retrievd successfully")

# =========================================================================================================================================================
# Normal tf-idf scoring & retrieval between a list of tokens in an inv_pos_index
def tf_idf(filtered_query,pos_index):
    score_dict = {}
    Vq_norm = 0

    # Calculating |Vq|
    for query in filtered_query:
        Vq_norm = Vq_norm + ((inv_pos_index[query][0])**2)
    Vq_norm = math.sqrt(Vq_norm)    

    # Calculating tf-idf scores
    for query in filtered_query:
        if(query in pos_index):
            doc_list = pos_index[query][1:]
            for doc in doc_list:
                tf = doc[1]
                idf  = pos_index[query][0]
                tf_idf = (idf * (tf*idf))/(Vq_norm * Vd_norm[doc[0]])
            
                if(doc[0] in score_dict):
                    score_dict[doc[0]] = score_dict[doc[0]] + tf_idf
            
                else:
                    score_dict[doc[0]] = tf_idf

    score_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    
    return score_dict[:10]

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# For Scoring in Local/Global Champion List
def champ(filtered_query, champ_dict):
    score_dict = {}
    Vq_norm = 0

    # Calculating |Vq|
    for query in filtered_query:
        Vq_norm = Vq_norm + ((inv_pos_index[query][0])**2)
    Vq_norm = math.sqrt(Vq_norm)    

    # Calculating |Vq|
    for query in filtered_query:
        doc_list = champ_dict[query]
        for doc in doc_list:
            tf = doc[1]
            idf  = inv_pos_index[query][0]
            tf_idf = (idf * (tf*idf))/(Vq_norm * Vd_norm[doc[0]])
        
            if(doc[0] in score_dict):
                score_dict[doc[0]] = score_dict[doc[0]] + tf_idf
        
            else:
                score_dict[doc[0]] = tf_idf
    score_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    
    return score_dict[:10]

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Part-IV Cluster Scoring Method
def cluster_scoring(filtered_query, follow_list, lead_inv_pos_index):
    score_dict = {}
    temp = tf_idf(filtered_query, lead_inv_pos_index)
    if(len(temp)==0):
        return []
    else:
        leader = temp[0]
    score_dict[leader[0]] = leader[1]
    
    followers = follow_list[leader[0]]

    # Calculating |Vq|
    Vq_norm = 0
    for query in filtered_query:
        Vq_norm = Vq_norm + ((inv_pos_index[query][0])**2)
    Vq_norm = math.sqrt(Vq_norm)    

    # Calculating tf-idf scores for follow_list
    for query in filtered_query:
            # Taking only the follower documents into consideration
            doc_list = [doc for doc in inv_pos_index[query][1:] if doc[0] in followers]
            for doc in doc_list:
                tf = doc[1]
                idf  = inv_pos_index[query][0]
                tidf = (idf * (tf*idf))/(Vq_norm * Vd_norm[doc[0]])
            
                if(doc[0] in score_dict):
                    score_dict[doc[0]] = score_dict[doc[0]] + tidf
            
                else:
                    score_dict[doc[0]] = tidf
    score_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    return score_dict[:min(10,len(score_dict))]
# -----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()