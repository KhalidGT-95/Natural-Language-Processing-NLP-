
## Name : Syed Khalid Ahmed
## Marticulation Number : 276970

import nltk
import numpy as np
from nltk.corpus import udhr
import math
import operator
from nltk.corpus import words
from nltk.corpus import brown

WordNgram = dict()
ReverseLookup = dict()
num_of_suggestions = None

## This function tokenizes a sentence and returns it
def Tokenizer(sentence):

    tokens = nltk.word_tokenize(sentence)
    return tokens

## Calculates the N-grams for given words in a list
def NgramCalculator(tokens):

    WordNgram = dict()  # Dictionary to store N-grams of a particular word
    weight = 3          # generate 3-grams

    for num_of_tokens in range(len(tokens)):    # Loop through all words in the list

        keyValue = dict()

        ## Find the N-grams for the word in the list
        for token_length in range(len(tokens[num_of_tokens])):  
            temp = tokens[num_of_tokens][token_length:token_length+weight]

            ## If the n-gram already occured previously in the given word
            if temp in keyValue:
                keyValue[temp] += 1     # Increase count by 1
            else:
                keyValue[temp] = 1      # If appearing for first time then assign a value 1
    
        WordNgram[tokens[num_of_tokens]] = keyValue     # Assign the N-gram dictionary to that particular word

    return WordNgram    # Return the Main Dictionary containing words with their N-grams


## Creates a dictionary which stores the N-gram as a key and the words in which
## that N-gram occurs as values.
## If 'th' appears in 4 words then the key will be 'th' and value will contain those 4 words
def aggregator(dataset):

    global ReverseLookup
    
    for parent_key, parent_value in dataset.items():

        for child_key, child_value in parent_value.items():
            
            if child_key in ReverseLookup:
    
                if parent_key in ReverseLookup[child_key]:
                    
                    ReverseLookup[child_key][parent_key] += child_value
                else:
                    
                    ReverseLookup[child_key].update({parent_key : child_value})                 
            else:
                ReverseLookup[child_key] = {parent_key : child_value}


## Check which of the N-grams of the input text appear 
def Comparator(dataset,word):
    global ReverseLookup
    
    PossibleCandidates = dict()

    for key, value in dataset[word].items():
        for chart_key,chart_value in ReverseLookup.items():
            if key == chart_key:
                for key, value in ReverseLookup[chart_key].items():

                    if key in PossibleCandidates:
                        pass
                    else:
                        PossibleCandidates[key] = value

    ## Pass the suggested words dictionary for calculating suggestions
    Final_suggestion(PossibleCandidates,word)
    
## Gives the final suggestions based on distance calculation
def Final_suggestion(PossibleCandidates,word):

    global num_of_suggestions

    distance_dict = dict()      # Dictionary to store the words and their edit distances
    
    for key, value in PossibleCandidates.items():   # Loop through all the possible candidates
        distance = edit_distance(key,word)          # Find the edit distance from each one of them

        distance_dict[key] = distance               # Put the suggested word with its edit distance

    ## Sort the dictionary based on edit distance
    sorted_dict = sorted(distance_dict.items(), key=operator.itemgetter(1))     

    ## Display the top k suggestions
    print("\nTop "+str(num_of_suggestions)+" suggestions for \""+word+"\" are : ")
    print([(k[0]) for k in sorted_dict[:num_of_suggestions]])


def substitution_error(b1,b2):
    if b1 == b2:
        return 0
    else:
        return 1

## This function calculates the Levenshtein Distance
def edit_distance(v, w):
    matrix = [[0 for j in range(len(w) + 1)] for i in range(len(v) + 1)]
    for i in range(len(v)+1):
        for j in range(len(w)+1):
            if i > 0 and j > 0:
                val1 = matrix[i-1][j] + 1
                val2 = matrix[i][j-1] + 1
                val3 = matrix[i-1][j-1] + substitution_error(v[i-1],w[j-1]) 
                matrix[i][j] = min(val1, val2, val3)
            elif i > 0:
                matrix[i][j] = matrix[i-1][j] + 1
            elif j > 0:
                matrix[i][j] = matrix[i][j-1] + 1
            else:
                matrix[i][j] = 0 


    return matrix[len(v)][len(w)]

## Splits the sentence, checks each individual word if it is already correct or not
## If it not a valid word then sends it for processing
def Sentence_splitting(sentence):

    words_set = set(words.words())  # Converts it into a set for faster lookup time
    
    text_tokens = sentence.split()  # Split the sentence

    ## Loop through all the tokens
    for i in range(len(text_tokens)):
        if text_tokens[i] in words_set:     # If it is a valid word
            print("\n\""+text_tokens[i]+"\""+" is a valid word")

        ## If not a valid word
        else:
            tokenized_values = Tokenizer(text_tokens[i])
            Ngrams = NgramCalculator(tokenized_values)
            Comparator(Ngrams,text_tokens[i])

## Takes input from the user the text string
def Take_input():

    ## First ask the user how many word suggestions he wants
    global num_of_suggestions
    num_of_suggestions = int(input("\nEnter the number of suggested words you want to be displayed : "))

    ## Loop until the user exits by pressing 'q'
    while True:
        print("\n\nEnter the string to check its words (Press 'q' to quit)")
        string = input("--> ")

        ## If 'q' pressed then program quits
        if string.strip().lower() == 'q':
            print("Goodbye :)")
            exit()

        ## Clean the text and send it for processing
        else:
            cleaned_text = string.strip().lower()
            Sentence_splitting(cleaned_text)
        
    
if __name__=="__main__":

    ## Used UDHR corpus for training
    print("\nTraining on UDHR corpus, Please Wait . . .")

    var = udhr.raw("English-Latin1")
    tokenized_values = Tokenizer(var.lower())
    Ngrams = NgramCalculator(tokenized_values)
    aggregator(Ngrams)

    print("\nTraining Done.")

    ## Used Gulliver's Travels for training as well to increase my training data set
    print("\nTraining on Gulliver's Travels book, Please Wait . . .")

    file = open("Gulliver.txt","r",encoding="utf-8")
    for line in file:
        tokenized_values = Tokenizer(line.strip().lower())
        Ngrams = NgramCalculator(tokenized_values)
        aggregator(Ngrams)

    print("\nTraining Done.")

    ## Starts taking input from the user
    Take_input()
