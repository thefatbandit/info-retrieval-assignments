import os 
import nltk 
from nltk.corpus import stopwords
from copy import deepcopy
import json

# Tokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
 
# Lemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def GeneratePermuterm(token):
    perm_list = []
    token = token + "$"
    for i in range(len(token)):
         perm_list.append(token)
         token = token[1:] + token[0]
    
    return perm_list

def GeneratePermutermList():
    perm_list = []
    with open("Inv_Pos_Index.json", 'r') as fp:
        inv_pos_index = json.load(fp)
        for token in inv_pos_index:
            perm_list.extend(GeneratePermuterm(token))
        
        perm_list.sort()
        return perm_list



def main():
    txt_file_path = os.getcwd() + "/ECTText/"
    stop_words = set(stopwords.words('english'))
    cust_stop_words = ["'s"]

    file_list = os.listdir(txt_file_path)

    # Adding custom words to the 
    for temp in cust_stop_words:
        stop_words.add(temp)

    inv_pos_index = {}

    # for i in range(1,3):
    for i in range(1,len(file_list)+1):
        txt_file_name = txt_file_path + "transcript-" + str(i) + ".txt"
        filtered_tokens = [] 
        with open(txt_file_name,"r") as text_file:
            text = text_file.read()
            
            # Tokenizing
            tokens = tokenizer.tokenize(text.lower())
                
            # Stop-Word Removal + Lemmatization
            for j in range(len(tokens)):
                temp = tokens[j]
                if(temp not in stop_words):
                    temp1 = lemmatizer.lemmatize(temp)
                    filtered_tokens.append((temp1,j)) # Storing their position in the original document as well for easier search

            #  Building Corpus
            for token in filtered_tokens:
                if(token[0] not in inv_pos_index):
                    token_dict = {}
                    pos_list = [token[1]]
                    token_dict[i] = pos_list
                    inv_pos_index[token[0]] = token_dict
                
                else:
                    token_dict = inv_pos_index[token[0]]
                    if(i not in token_dict.keys()):
                        pos_list = [token[1]]
                        token_dict[i] = pos_list
                        inv_pos_index[token[0]] = token_dict
                    else:
                        token_dict[i].append(token[1])
            
            if(i%50==0):
                print(str(i) + " | " + str(len(inv_pos_index.keys())))

    print(str(len(file_list)) + " | " + str(len(inv_pos_index.keys())))

    # Storing of Postings List
    with open("Inv_Pos_Index.json", 'w') as fp:
        json.dump(inv_pos_index, fp, sort_keys=True, indent=3)

    # Generating & Storing Permuterm
    perm_list = GeneratePermutermList()
    with open("Permuterm.txt","wt") as text_file:
        text_file.write(" ".join(perm_list))
    
if __name__ == "__main__":
    main()