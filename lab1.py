###############################importing libraries######################################
import sys
import glob
import os
import re
from collections import Counter
from random import shuffle
import random
import numpy
import matplotlib.pyplot as plt
n=0
#######################################Cmd line Code########################################
Path =  sys.argv[1]
Path_neg = Path+'/txt_sentoken/neg'
Path_pos = Path+'/txt_sentoken/pos'

###################################declearing global variable###############################################
pos_count=[] #having positive document and the key values of word.
neg_count=[] #having negative document and the key values of word..
weights={} #weights of all words.
x_train =[] #It cantains 1600 document(800 positive and 800 negative).
x_test=[] #It cantains 400 document(200 positive and 200 negative).
a_weights=[] #list of weights returned after training model.

###################################function###########################################   
def read_tokenisation_counter(n): #read file and then perform tokenisation by calling function 'ngarms'
    input_file = [] #list in which all documents are saved.
    file_lists = glob.glob(os.path.join('review_polarity/txt_sentoken/pos/', '*.txt')) #loading positive document from folder.
    for file_list in file_lists: # for loop to read each document one by one.
        with open(file_list,'r') as f_input: #reading document using syntax 'r'.
            input_file = f_input.read() #storing file in list.
            bais = 'baisnlp' #adding unique word to perform bais function.
            input_file = input_file+" "+bais #adding word to eacj and evry document.
            dictionary = Counter(ngrams(input_file, n)) #Counting words occurence per document.
            pos_count.append(dictionary) #collection of count of words of positive document.
    file_lists = glob.glob(os.path.join('review_polarity/txt_sentoken/neg/', '*.txt')) #loading negative document from folder.
    for file_list in file_lists: # for loop to read each document one by one.
        with open(file_list,'r') as f_input: #reading document using syntax 'r'.
            input_file= f_input.read() #storing file in list.
            bais = 'baisnlp'  #adding unique word to perform bais function.
            input_file = input_file+" "+bais #adding word to eacj and evry document.
            dictionary = Counter(ngrams(input_file, n)) #Counting words occurence per document.
            neg_count.append(dictionary) #collection of count of words of negative document.

def ngrams(input_file,n):
    input_file = input_file.lower() # converting files into lower case.
    input_file = re.sub(r'[^a-zA-Z0-9\s]', ' ', input_file) # regex fution to tokenise text for unigram,bigram and tigram.
    tokenisation = [token for token in input_file.split(" ") if token != ""] #breaking file into token and removing empty spaces.
    ngrams = zip(*[tokenisation[i:] for i in range(n)]) #generation of n garms.
    return [" ".join(ngram) for ngram in ngrams] #concating token with ngarms and returning them.

def train_model(pos_count,neg_count):
    for x in pos_count[:800]:  #taking first 800 positive document into x_train.
        x_train.append([x,1])   
    for x in pos_count[800:]: #taking left 200 positive document into x_test.
        x_test.append([x,1])
    for x in neg_count[:800]: #taking first 800 negative document into x_train.
        x_train.append([x,-1])
    for x in neg_count[800:]: #taking first 200 negative document into x_test.
        x_test.append([x,-1])

def weight(x_train): # assigning zero weights to the each and evry word in document.
    for i in range(len(x_train)):
        for word in x_train[i][0]:
            weights[word]=0.0
    return weights

#########################################Testing Model#############################################

def binary_perceptron_theoram_test(weights,t): # taking return weight and testing document in argument.
    correct=1  #setting counter.
    for i in range(len(t)):# running for loop to find score over the length of test documents'400'.
        score=0.0 #initailized varible.
        for words in t[i][0]: #running 'for' loop on x_test,'t[i][0]' this chosing document from x_test. 
            if words not in weights: # condition for loop to continue if the word is not found in x_test.
                continue
            score+=t[i][0][words] * weights[words]
        if score >= 0: # if the score is greater then 1 than 0 then it adds word to the counter.
            if t[i][1]==1:
                correct+=1
        else :
            if t[i][1]==-1:
                correct+=1
    return (correct/len(t)) # finding accuracy by dividing the sum of words in counter 'correct' with the count of document in 'x_test'.

############################################Standard Training Model#################################################################
def standard_binary_perceptron_theoram_without_shuffling(x_train,weights,x_test): # Function to find the standard accuracy of theoram on the training data.
   for dictionary,y in x_train: #for loop on x_train(which containg both document and the label.)
       score = 0
       for word in dictionary: # iterating for loop on document in x_train for words in document.
           score+= dictionary[word] * weights[word]
       y_haat = numpy.sign(score)
       if y_haat!=y:
           if y==1:
              for x in dictionary:   #if y=1 then it adds weights otherwise it substract it.
                  weights[x]+=dictionary[x]
           else:
              for x in dictionary:
                  weights[x]-=dictionary[x]
   Accuracy=binary_perceptron_theoram_test(weights,x_test) # calling "binary_perceptron_theoram_test" function to find the accuracy.
   print("Accuracy at standard training data without shuffling",Accuracy)
   return weights

   
############################################Standard Training Model with shuffling###################################################
def standard_binary_perceptron_theoram(x_train,weights,x_test): # Function to find the standard accuracy of theoram on the training data.
    shuffle(x_train) #shuffling document in x_train.
    for dictionary,y in x_train: #for loop on x_train(which containg both document and the label.)
        score = 0
        for word in dictionary: # iterating for loop on document in x_train for words in document.
            score+= dictionary[word] * weights[word]
        y_haat = numpy.sign(score)
        if y_haat!=y:
            if y==1:
                for x in dictionary:   #if y=1 then it adds weights otherwise it substract it.
                    weights[x]+=dictionary[x]
            else:
                for x in dictionary:
                    weights[x]-=dictionary[x]
    Accuracy=binary_perceptron_theoram_test(weights,x_test) # calling "binary_perceptron_theoram_test" function to find the accuracy.
    print("Accuracy at standard training data with shuffling",Accuracy)
    return weights

############################################Training Model####################################################            

def binary_perceptron_theoram_training(x_train,weights,x_test,n): # Function to find the standard accuracy of theoram on the training data.
    error_list=[] #declaring error_list(array) to save error for each training document.
    for i in range(20):# iterating it 16 times to train model.
        shuffle(x_train) #shuffling training data 16 times.
        for dictionary,y in x_train: #for loop on x_train(which containg both document and the label.)
            score = 0
            for word in dictionary: # iterating for loop on document in x_train for words in document.
                score+= dictionary[word] * weights[word]
            y_haat = numpy.sign(score)
            if y_haat!=y:
                if y ==1: #if y=1 then it adds weights otherwise it substract it.
                    for x in dictionary:
                        weights[x]+=dictionary[x]
                        
                else:
                    for x in dictionary:
                        weights[x]-=dictionary[x]
        a_weights.append(weights.copy()) #
        Accuracy=binary_perceptron_theoram_test(weights,x_test)# calling "binary_perceptron_theoram_test" function to find the accuracy.
        error=binary_perceptron_theoram_test(weights,x_train) #  find error from training data.
        print("Accuracy at iteration", i ,"is",Accuracy)
        error_list.append(1-error) #appending error calculate into the array.
#####################################Graph##################################################        
    # Title for plot.
    if(n == 1):
        plt.title("Unigram Error Plot")
    elif(n == 2):
        plt.title("Bigram Error Plot")
    else:
        plt.title("Trigram Error Plot")
    plt.plot(error_list) # ploting graph for error_list.
    plt.ylabel('Error') #labelling 
    plt.xlabel('Iteration')
    plt.show()
    return weights
    
####################################Averge Weight###########################################                   
def averge_weight_accuracy(weights): #taking averge of weight.
    summation = Counter() 
    count = Counter()
    for dictionary in weights: #'for' loop on weights for dictionary. 
        summation.update(dictionary) #updating 
        count.update(dictionary.keys())
        averge_weight = {x: float(summation[x])/count[x] for x in summation.keys()}
    Accuracy=binary_perceptron_theoram_test(averge_weight,x_test)
    print(" Averge Accuracy is",Accuracy)
    return averge_weight
#############################Top 10 negative and positive words#############################
def top_10_neg_pos_word(av_weight): # finding top 10 neagtive and positve word.
    positive_words=sorted(av_weight.items(), key=lambda kv:-kv[1]) #descending sorting.
    print("Top 10 Positive word",positive_words[:10])
    negative_words=sorted(av_weight.items(), key=lambda kv:kv[1]) ##ascending sorting.
    print("Top 10 Negative word",negative_words[:10])
#################################Input functions for Unigram,Bigram and Trigram ###########################################  
def intialize(): #Function taking input form user for unigram, bigram and trigram.
    global n
    seed_val = input("Enter random seed value : ") #taking input for random seed.
    random.seed(seed_val)
    input_value= input("Enter the value of ngram: ")
    n=int(input_value)
    if input_value==str(1):
        read_tokenisation_counter(1)
    elif input_value==str(2):
        read_tokenisation_counter(2)
    else:
        read_tokenisation_counter(3)
########################################Callimg Functions#########################################        

intialize()       
train_model(pos_count,neg_count)
weight(x_train)
standard_binary_perceptron_theoram_without_shuffling(x_train,weights,x_test)
standard_binary_perceptron_theoram(x_train,weights,x_test)
training_model=binary_perceptron_theoram_training(x_train,weights,x_test,n)
test_model=binary_perceptron_theoram_test(training_model,x_test)
av_weight=averge_weight_accuracy(a_weights)
top_10_neg_pos_word(av_weight) 


