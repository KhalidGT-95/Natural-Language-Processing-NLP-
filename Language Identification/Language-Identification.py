
## Name : Syed Khalid Ahmed
## Marticulation number : 276970

import numpy as np
import nltk
from nltk.corpus import udhr
import math
from decimal import *
import re

udhr_store_english = dict()
udhr_store_german = dict()
udhr_store_italian = dict()
udhr_store_spanish = dict()

weight = None
stores = [udhr_store_english,udhr_store_german,udhr_store_italian,udhr_store_spanish]
names = ['English','German','Italian','Spanish']

def NgramCalculator(text,choice):

    input_store = dict()
    global stores
    global names
    global weight
    
    # Taking 2 grams
    weight = 2

    if choice == 0:                          # For Input 
        for i in range(len(text)):           # Loop until the length of the text   
            temp = text[i:i+weight].strip().lower()     # Produce n-grams, strip the whitespaces and convert into lowercase

            if temp in input_store:     # If the n-gram is already in the store    
                input_store[temp] += 1  # Increment its count by 1

            else:                       # If appearing for the first time
                input_store[temp] = 1   # Create a key of it and assign a value of 1

        CosineSimilarity(input_store)   # Call this function to find the Cosine Similarity
        
    else:                               # For language corpora
        for i in range(len(text)):
            
            temp = text[i:i+weight].strip().lower()

            ## The following lines perform text cleanup by removing special characters,
            ## numbers, tabs and newline characters from the text. Since these do not
            ## help in language identification, so it is better to remove them. 
            ## Now the n-grams contain values which are of most interest. 

            temp = re.sub('[,!@#$-]','',temp)        
            temp = re.sub('[0-9]','',temp)
            temp = re.sub('\t',' ',temp)
            temp = re.sub('\n',' ',temp).strip()

            #########################################################
            
            # Since the stores is a list and each on each index is a dictionary
            # so we can perform a dictionary lookup
            
            if temp in stores[choice-1]:            
                stores[choice-1][temp] += 1
            else:
                stores[choice-1][temp] = 1

def Lookup(key,dict_index):
    global stores
    global names

    for i in range(len(stores)):    # Loop through the available dictionaries    
        if i == dict_index:         # If on the same dictionary as the language, skip it since we are interested in finding the n-gram in other dictionaries
            continue
        
        if key in stores[i]:        # If the n-gram is also present in another dictionary, then it is common among languages and hence of no interest
            return False            # immediately return false

    return True                     # If not present in any other dictionary, return true

def CosineSimilarity(input_store):

    global stores
    global names
    global weight

    print("\t\t\t\t\tStatistics :\n")

    ## Loop through all the number of dictionaries present    
    ## Since there are 4 dictionaries, so this loop will run from 0-3
    for i in range(len(stores)):
        numerator = 0           # Numerator = 0 for each iteration             
        denominator = 0         # Numerator = 0 for each iteration
        TrainingData_temp = 0   # Training data values
        TestingData_temp = 0    # Testing data values

        ## Loop through each key-value pair in the input n-gram dictionary
        for key,value in input_store.items():

            # If the key is present in the language dictionary    
            if key in stores[i]:
                if Lookup(key,i):   # Perform a lookup on other dictionaries, if it is true then
                    numerator += (value * int(stores[i][key])) ** 2     # Raise the power of numerator by 2 since it is highly probably that the current language is same as input
                    TrainingData_temp += value**2
                    TestingData_temp += int(stores[i][key]) ** 2

                else:
                    numerator += (value * int(stores[i][key]))
                    TrainingData_temp += value**2
                    TestingData_temp += int(stores[i][key]) ** 2
            else:
                numerator += (value * 0)
                TrainingData_temp += value**2
                TestingData_temp += 0
                
        denominator = math.sqrt(TrainingData_temp) * math.sqrt(TestingData_temp)

        try:
            cos_theta = numerator / denominator
            if cos_theta >= 0.99:       # If the score reaches above 99%
                cos_theta = 0.99        # Clip it to 99%

            print("For " + names[i] + ", similarity percentage is: ")
            print(str(round(cos_theta*100,3)) + " % \n")

        except ZeroDivisionError:       # Thrown when no word matches. For Example : süß will never appear in english
            print("No matching word in "+names[i])
    
def TakeInput():

    while True:
        print("Please enter a string to find its language similarity (press 'q' to quit) \n")
        string = str(input("--> "))

        if string.strip() == 'q':
            print("\nGoodbye :)")
            break

        else:
            NgramCalculator(string.strip(),0)   # Pass this to N-gram function with code 0 specifying this as an input 
            print("____________________________________________________\n")

def Training():

    global weight
    
    print("\nTraining the model using the given data , Please Wait . . . \n")

    ## Read the corpora
    english = udhr.raw("English-Latin1")
    german = udhr.raw("German_Deutsch-Latin1")
    italian = udhr.raw("Italian-Latin1")
    spanish = udhr.raw("Spanish-Latin1")

    ## Pass these to NgramCalculator to calculate n-grams
    NgramCalculator(english,1)
    NgramCalculator(german,2)
    NgramCalculator(italian,3)
    NgramCalculator(spanish,4)

    print("Taking "+str(weight)+" grams")

    ## Read the news files sequentially 
    for i in range(len(names)):
        filename = names[i]+".txt"
        string = ""
        with open(filename,encoding="utf-8") as file:
            content = file.readlines()
            for line in content:
                string += "".join(line)     # Append to the string
        
        NgramCalculator(string,i+1)


    print("\nTraining Completed . . .\n")
    
if __name__=="__main__":

    Training()    

    TakeInput()
