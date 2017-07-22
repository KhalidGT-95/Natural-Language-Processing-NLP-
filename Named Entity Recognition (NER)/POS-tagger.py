#### Name : Syed Khalid Ahmed
#### Marticulation number : 276970

#!/usr/bin/env python

import nltk
from nltk.corpus import conll2000 
from pickle import dump

## Here I have implemented a combined tagger to increase the accuracy of the tagger
def BackoffTaggers(train_data,test_data):

    Default_tagger = nltk.DefaultTagger('NN')
    Unigram_tagger = nltk.UnigramTagger(train_data, backoff=Default_tagger)
    Bigram_tagger = nltk.BigramTagger(train_data, backoff=Unigram_tagger)

    print("\nAccracy on the test data comes out to be : ",end='')
    print(Bigram_tagger.evaluate(test_data))  # Evaluating the accuracy on test data  
    
    ## Dump the tagger in a file to be used later
    output = open('tagged_data.pkl','wb')
    dump(Bigram_tagger,output, -1)
    output.close()
    
    print("\nDumped the data in a file for later use")

## Function to read the CONLL corpus
def read_conll(path):
    result = []
    file = open(path)
    sent = []
    for line in file:
        line = line.strip('\n')
        if not line.strip(' '):
            result.append(sent)
            sent = []
            continue
        (word,pos,tag) = line.split(' ')
        sent.append((word,pos))     # storing only word and POS to train the tagger
    return result

if __name__=="__main__":
    

    conll_train = read_conll('train.txt')   # Read the CONLL training text
    conll_test = read_conll('test.txt')     # Read the CONLL testing text

    BackoffTaggers(conll_train,conll_test)  # Call the function
    
