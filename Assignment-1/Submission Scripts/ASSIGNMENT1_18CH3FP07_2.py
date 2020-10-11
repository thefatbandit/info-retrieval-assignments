from bs4 import BeautifulSoup
import os 
import re
import nltk
from copy import deepcopy
import json

def CreateNestedDict(file_list, file_path):
    transcript = {}
    debug_list = []
    # Dictionary Construction
    for i in range(len(file_list)):
        path = file_path + file_list[i]
        ID = re.findall(r'\d+', file_list[i])[0]
        # print("DocID: " + ID)

        tr_file = open(path)
        # Parsing the html
        soup = BeautifulSoup(tr_file, features="html.parser")

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        #  --------------------------------------------------------------------------------------------------------------
        # Finding & Storing Date
        # Storing the 1st line
        a = (soup.find('p', {'class' : 'p p1'})).get_text()  # For finding a specific class in the html
        #  Finding the date using regex
        a = re.findall(r'[A-Z]\w+\s+\d+\,\s+\d+',a)

        
        # Brute-Forcing the "p p_1" class in case Dat not in heading
        if(not a): 
            temp = soup.findAll('p', class_="p p1")
            for j in range(5): # Checking in the 1st 5 tags
                a = re.findall(r'[A-Z]\w+\s+\d+\,\s+\d+', temp[j].get_text())
                if(not a):
                    a = re.findall(r'[A-Z]\w+\s+\d+\s+\d+', temp[j].get_text())
                if(a):
                    break
        if(not a):
            print(temp[j].get_text())
        # Removing extra spaces from the date
        date = " ".join(a[0].split())

        #  --------------------------------------------------------------------------------------------------------------
        # Constructing Participant List
        a = soup.findAll('p')
        part_list= []
        flag=0  # flag for storing no. of strong texts <strong> texts
        ctr=-1

        while(flag<3):
            ctr+=1
            tag = a[ctr]
            if(tag.find('strong')):
                flag+=1
            elif(flag>0):
                part_list.append(tag.get_text())

        #  --------------------------------------------------------------------------------------------------------------
        # Constructing Presentation Dictionary
        presentation_dict ={}
        temp =[]
        speaker = ""
        flag = 0
        qna_head_phrases = ["question-and-answer session", "question-and-answers session", "questions-and-answers session", "questions-and-answer session"]

        while(ctr<len(a) and (" ".join(((a[ctr].get_text()).lower()).split()) not in qna_head_phrases)):
            tag = a[ctr]
            if(tag.find('strong')):
                if((tag.get_text() not in presentation_dict.keys())):
                    if(flag==0):
                        speaker = tag.get_text()
                        flag=1
                    elif(flag!=0):
                        presentation_dict[speaker] = temp
                        temp=[]
                        speaker = tag.get_text()
                else:
                    presentation_dict[speaker] = temp
                    speaker = tag.get_text()
                    temp = presentation_dict[speaker]
            else:
                temp.append(tag.get_text())
            ctr+=1
        else: # To add the last speaker to presentation_dict
            presentation_dict[speaker] = temp
        
        if(ctr==len(a)):
            debug_list.append(ID)

        #  --------------------------------------------------------------------------------------------------------------
        # Construction of Questionnaire Dictionary
        ctr+=1 # To move forward from the Q-&-A Heading tag
        question_dict = {}
        temp = {}
        speaker = ""
        flag = 0
        remarks = []
        flow_ctr= 1
        while(ctr<len(a)): # Looping till end of the document
            tag =a[ctr]
            if((tag.find('strong') or (tag.get_text() in part_list))):
                if(flag==0):
                    speaker = tag.get_text()
                    flag=1
                else:
                    temp["Speaker"] = deepcopy(speaker)
                    temp["Remarks"] = remarks
                    question_dict[flow_ctr] = deepcopy(temp) # Important to use deepcopy
                    speaker = tag.get_text()
                    remarks = []
                    temp = {}
                    flow_ctr+=1
            else:
                remarks.append(tag.get_text())
            ctr+=1
        else: # To add the last speaker to question_dict. 
            temp["Speaker"] = deepcopy(speaker)
            temp["Remarks"] = remarks
            question_dict[flow_ctr] = deepcopy(temp)

        #  --------------------------------------------------------------------------------------------------------------    
        # Creating the Transcript Instance 
        transcript["Date"] = date
        transcript["Presentation"] = presentation_dict
        transcript["Questionnaire"] = question_dict

        #  --------------------------------------------------------------------------------------------------------------    
        # Saving the transcript Dictionary in a JSON file
        script_file_path = os.getcwd() + "/ECTNestedDict/"
        json_name = "transcript-" + str(i+1) + ".json"
        with open(script_file_path + json_name, 'w') as fp:
            json.dump(transcript, fp, sort_keys=True, indent=3)

def CreateText(file_path,txt_file_path):
    file_list = os.listdir(file_path)
    debug_list = []


    for i in range(1,len(file_list)+1):
        # JSON + TXT file name for Read/Write 
        json_file_name = file_path + "transcript-" + str(i) + ".json"
        txt_file_name = txt_file_path + "transcript-" + str(i) + ".txt"
        text = ""

        # Opening JSON file 
        with open(json_file_name, 'r') as transcript_file:
            transcript = json.load(transcript_file)
            presentation_dict = transcript["Presentation"]
            for presenter in presentation_dict:
                remarks = presentation_dict[presenter]
                if(remarks):
                    text = text + remarks[0]
                for j in range(1, len(remarks)):
                    text = text + " " + remarks[j]
            question_dict = transcript["Questionnaire"]
            for speaker in question_dict:
                remarks = question_dict[speaker]["Remarks"]
                for temp in remarks:
                    text = text + " " + temp
        with open(txt_file_name,"wt") as text_file:
            text_file.write(text)


def main():
    file_path = os.getcwd() + "/ECT/"
    
    # Making "ECTNestedDict" directory
    script_file_path = os.getcwd() + "/ECTNestedDict/"
    os.mkdir(script_file_path)
    #  Listing out all files in "ECT"
    file_list = os.listdir(file_path)
    
    # Funciton for Task 2.1
    CreateNestedDict(file_list, file_path)

    # JSON file location
    file_path = os.getcwd() + "/ECTNestedDict/"
    # Creating "ECTText" folder
    txt_file_path = os.getcwd() + "/ECTText/"
    os.mkdir(txt_file_path)
    
    # Function for Task 2.2
    CreateText(file_path, txt_file_path)

if __name__ == "__main__":
    main()
    