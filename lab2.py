
###################################################Importing Libraries###################################################################
import re
import sys
#####################################################Global Variable#####################################################################
n = 0 # intialised variable
train_data = sys.argv[1] # Training data path.
test_data = sys.argv[2] # Test data path.
length_of_unigram_dict=0 # total no of unigram words.
dict_bigram=0
correct_words = ['whether', 'through', 'piece', 'court', 'allowed', 'check', 'hear', 'cereal', 'chews', 'sell'] #correct unigram words
correct_words_bigram = ['know whether', 'went through', 'a piece', 'to court', 'only allowed', 'to check', 'you hear', 'eat cereal', 'normally chews','to sell'] #correct bigram words
dict_prob_unigram = {} #unigram probabilities dictionary.
dict_prob_bigram = {} #bigram probabilities dictionary.
d = {} #dictionary with word and their frequency.

########################################################Function#########################################################################

def ngrams(n, data):
    fo = open(data, "r") #opening file
    input_file = fo.readlines() #reading lines of files.
    for i in range(len(input_file)): #reading line one by one.
        letters = input_file[i].lower() #lowering each character
        if (n == 2): #condition for the  bigram model.
            letters = "<s>" + " " + letters + " " + "</s>" # appending <s> and </s> in the start and end.
        regex = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(letters)) #regex function to split words.
        tokenisation = [token for token in regex.split(" ") if token != ""]
        ngrams = zip(*[tokenisation[i:] for i in range(n)])
        concate = [" ".join(ngram) for ngram in ngrams]
        #dictionary.update({'S_' + str(i + 1): concate})
        [d.__setitem__(item, 1 + d.get(item, 0)) for item in concate] #dictionary with word and their frequency.
    return (d)

def Word_Probabilities(n): #predicting probability of each word using this function.
    if n==1: # Unigram
        length_of_unigram_dict = sum(d.values()) # length of dictionary
        for key, value in sorted(d.items()):
            probability = value / length_of_unigram_dict # probability of each word
            dict_prob_unigram[key] = probability # unigram probabilities dictionary.
    else:
        dict_bigram = sum(d.values()) # length of dictionary
        for key, value in sorted(d.items()):
            probability = value #/ dict_bigram # probability of each word
            dict_prob_bigram[key] = probability # bigram probabilities dictionary.


def prediction_unigram(data):
    value = 0 # initialised variable.
    fo = open(data, "r") # opening test file
    input_test_file = fo.readlines() # reading test file
    for i in range(len(input_test_file)):
        letters = input_test_file[i].lower() # lowering the characters of input file.
        given_word_1 = letters.split(':')[1].split('/')[0].strip() # splitting words
        given_word_2 = letters.split(':')[1].split('/')[1].strip() # splitting words
        aaa = predict(given_word_1, given_word_2) #calling functions
        if (aaa == correct_words[i]): # Calculating accuracy.
            value = value + 1 #counting correct words
    accuracy = value / 10
    print("Accuracy of Unigram Model =",accuracy)


def predict(word1, word2):
    prob1 = dict_prob_unigram[word1] # prob of first word in unigram.
    prob2 = dict_prob_unigram[word2] # prob of second word in unigram.
    if prob1==prob2:
        return "prob_is_same"
    if prob1 > prob2:  #comparing values of prob1 and prob2 to find the word with higher probability.
        return (word1)
    else:
        return (word2)


def prediction_bigram_1(data): # function for calculating bigram.
    accuracy=[]
    count=0 # initailize variable
    fo = open(data, "r") #opening file.
    input_test_file = fo.readlines() #reading file.
    for i in range(len(input_test_file)):
        letters = input_test_file[i].lower() # lowering the characters of input file.
        given_word_1 = letters.split(':')[1].split('/')[0].strip() # prob of first word in bigram.
        given_word_2 = letters.split(':')[1].split('/')[1].strip() # prob of second word in bigram.
        previous_word = letters.split('____')[0].split()[-1] # prob of previous word.
        next_word = letters.split('____')[1].split()[0] #next word after the space.
        w4_1 = given_word_1 + " " + next_word #Adding 'next word' and 'word to predicted
        w5_1 = given_word_2 + " " + next_word #Adding 'next word' and 'word to predicted
        w4 = previous_word + " " + given_word_1 #Adding 'previous word' and 'word to predicted'
        w5 = previous_word + " " + given_word_2 #Adding 'previous word' and 'word to predicted'
        #for previous and predicted word.
        if w4 not in dict_prob_bigram:  #if word4 not in bigram dictionary then assign probability 'zero' or will calculate probability of 'word_4'.
            pw4 = 0
        else:
            pw4 = dict_prob_bigram[w4] / dict_prob_unigram[previous_word]
        if w5 not in dict_prob_bigram: #if word5 not in bigram dictionary then assign probability 'zero' or will calculate probability of 'word_5'.
            pw5 = 0
        else:
            pw5 = dict_prob_bigram[w5] / dict_prob_unigram[previous_word]
        # for next and predicted word.
        if w4_1 not in dict_prob_bigram:  # if word4 not in bigram dictionary then assign probability 'zero' or will calculate probability of 'word_4'.
            pw4_1 = 0
        else:
            pw4_1 = dict_prob_bigram[w4_1] / dict_prob_unigram[given_word_1]
        if w5_1 not in dict_prob_bigram:  # if word5 not in bigram dictionary then assign probability 'zero' or will calculate probability of 'word_5'.
            pw5_1 = 0
        else:
            pw5_1 = dict_prob_bigram[w5_1] / dict_prob_unigram[given_word_2]
         #mulitplying the probabilities of previous,given and next word.
        pw4 = pw4 * pw4_1
        pw5 = pw5 * pw5_1

        if pw4==pw5: # to detect term which have same prob then append nothing.
            accuracy.append('eeeee')
        else:
            if pw4==pw5: # to detect term which have same prob then append nothing.
                accuracy.append("ERROR")
            elif pw4 > pw5: #comparing values of prob1 and prob2 to find the word with higher probability.
                accuracy.append(w4)
            else:
                accuracy.append(w5)
            if (accuracy[i] == correct_words_bigram[i]):  # Calculating accuracy.
                count = count + 1  # counting correct words
    acc = count / 10

    print("Accuracy of Bigram Model =",acc)

def prediction_bigram_smoothing(data): # function for calculating bigram model..
    accuracy=[]
    count=0
    fo = open(data, "r") #opening file.
    input_test_file = fo.readlines() #reading file.
    for i in range(len(input_test_file)):
        letters = input_test_file[i].lower() # lowering the characters of input file.
        given_word_1 = letters.split(':')[1].split('/')[0].strip() # prob of first word in bigram.
        given_word_2 = letters.split(':')[1].split('/')[1].strip() # prob of second word in bigram.
        previous_word = letters.split('____')[0].split()[-1] # prob of previous word.
        next_word = letters.split('____')[1].split()[0]#next word after the space.
        w4_1 = given_word_1 + " " + next_word #Adding 'next word' and 'word to predicted
        w5_1 = given_word_2 + " " + next_word  #Adding 'next word' and 'word to predicted
        w4 = previous_word + " " + given_word_1 #Adding 'previous word' and 'word to predicted'
        w5 = previous_word + " " + given_word_2 #Adding 'previous word' and 'word to predicted
        # for previous and predicted word.
        if w4 not in dict_prob_bigram: # if 'word_4' is not in the dictionary then probability be
            pw4 = (1)/(dict_prob_unigram[previous_word] + dict_bigram) # Implementing laplace function.
        else:
            pw4 = (dict_prob_bigram[w4]+1)/(dict_prob_unigram[previous_word] + dict_bigram)
        if w5 not in dict_prob_bigram:
            pw5 = (1)/(dict_prob_unigram[previous_word] + dict_bigram)
        else:
            pw5 = (dict_prob_bigram[w5]+1)/(dict_prob_unigram[previous_word] + dict_bigram)
        # for next and predicted word.
        if w4_1 not in dict_prob_bigram: # if 'word_4' is not in the dictionary then probability be
            pw4_1 = (1)/(dict_prob_unigram[given_word_1] + dict_bigram) # Implementing laplace function.
        else:
            pw4_1 = (dict_prob_bigram[w4_1]+1)/(dict_prob_unigram[given_word_1] + dict_bigram)
        if w5_1 not in dict_prob_bigram:
            pw5_1 = (1)/(dict_prob_unigram[given_word_2] + dict_bigram)
        else:
            pw5_1 = (dict_prob_bigram[w5_1]+1)/(dict_prob_unigram[given_word_2] + dict_bigram)
        # mulitplying the probabilities of previous,given and next word.
        pw4 = pw4 * pw4_1
        pw5 = pw5 * pw5_1
        if pw4==pw5: # to detect term which have same prob then append nothing.
            accuracy.append('xxx')
        else:
            if pw4==pw5: # to detect term which have same prob then append nothing.
                accuracy.append("ERROR")
            elif pw4 > pw5: #comparing values of prob1 and prob2 to find the word with higher probability.
                accuracy.append(w4)
            else:
                accuracy.append(w5)
        if (accuracy[i] == correct_words_bigram[i]):  # Calculating accuracy.
            count = count + 1  # counting correct words
        acc = count / 10
    print("Accuracy of Bigram Model with smoothing =",acc)


########################################################Calling Function########################################################################
ngrams(1, train_data) #unigram function
Word_Probabilities(1) #Finding probabilities
prediction_unigram(test_data) #For Unigram Model
ngrams(2, train_data) #unigram function
Word_Probabilities(2) #Finding probabilities
prediction_bigram_1(test_data) #For Bigram Model
prediction_bigram_smoothing(test_data) #For Bigram Model with smoothing
