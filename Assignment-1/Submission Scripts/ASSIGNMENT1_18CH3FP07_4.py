# Imports Required
import json
import sys

# Loading Stored Permuterms
with open("Permuterm.txt","r") as permuterm_file:
    text = permuterm_file.read()
    perm_list = text.split()

# Loading Posting List
with open("Inv_Pos_Index.json", 'r') as fp:
    inv_pos_index = json.load(fp)

def FindRange(token, pos):
    low_bound = pos
    high_bound = pos
    
    while((perm_list[low_bound].startswith(token)) and (low_bound > 0)):
      low_bound = low_bound - 1
    
    while((perm_list[high_bound].startswith(token)) and (high_bound < len(perm_list))):
      high_bound = high_bound + 1
    
    return(low_bound+1, high_bound-1)

def TokenSearch(token, start, end):
    # print(str(start) + " | " + str(end))
    if(start==end):
        if(perm_list[start].startswith(token)):
            return (start, start)
        else:
            return -1
    if(start>end or (end - start == 1)):
        return -1

    mid = int((start + end)/2)
    if(perm_list[mid].startswith(token)):
        token_range = FindRange(token,mid)
    else:
        if(token < perm_list[mid]):
            token_range = TokenSearch(token, start, mid-1)
        else:
            token_range = TokenSearch(token, mid, end)
    return token_range

def QueryToToken(query):
    # mon* CASE
    if(query[-1] == "*"):
        return ("$" + query[:-1])
    # *mon CASE
    elif(query[0] == "*"):
        return (query[1:] + "$")
    # mon*mon CASE
    else:
        star_index = query.find('*')
        return(query[star_index+1:] + "$" + query[:star_index])

def TokenToWord(token):
    if(token[-1]=='$'):
        return(token[:-1])
    else:
        dollar_index = token.find("$")
        return(token[dollar_index+1:] + token[:dollar_index])

def main():
    # Sotring all queries in a list
    with open(sys.argv[1],"r") as query_file:
        query_list = query_file.readlines()

    text_list = []    
    with open("RESULTS1_18CH3FP07.txt","wt") as text_file:
        for query in query_list:
            # Generating Token for Query
            token = QueryToToken(query.rstrip("\n"))

            # Searching for approriate range for the query given
            token_range = TokenSearch(token, 0, len(perm_list)-1)

            if(token_range == -1):
                text_file.write("Query Not Found")
                text_file.write("\n")

            else:
                token_list = []
                for i in range(token_range[0],(token_range[1]+1)):
                    token_list.append(TokenToWord(perm_list[i]))

                text = ""
                for token in token_list:
                    text+= token +":"
                    token_dict = inv_pos_index[token]
                    for doc in token_dict:
                        posting_list = token_dict[doc]
                        for posting in posting_list:
                            text+= "<" + str(doc) + "," + str(posting) + ">,"
                    text = text[:-1]
                    text+= ";"
                text_file.write(text[:-1])
                text_file.write("\n")

if __name__ == "__main__":
    main()