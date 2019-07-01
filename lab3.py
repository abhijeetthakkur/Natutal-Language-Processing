############################################################# importing libraries ###########################################################################################
import itertools
from collections import Counter
from sklearn.metrics import f1_score
import random
import copy
import sys

random.seed(1234)


############################################################# Decleration Of Variable#########################################################################################
train_file = sys.argv[1] # Training data path.
test_file = sys.argv[2] # Test data path.

# train_file = "train.txt" # training file.
# test_file = "test.txt" # testing file.
tag_list = ["O","PER","LOC","ORG","MISC"] #tags given.

############################################################ Functions #######################################################################################################
##Below function provided is producing training data.
###############################################################################################################################################################
def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets) #function returining the tuple.
###############################################################################################################################################################
##It is providing us with word and tag count.
###############################################################################################################################################################
def Extraction_of_feature(data):
    merged_list = list(itertools.chain.from_iterable(data))
    frequency_of_words = Counter(merged_list) # merging list.
    cw_cl_counts = {} # dictionary for the words with frequency.
    for key, values in frequency_of_words.items():
        cw_cl_counts.update({key[0] + "_" + key[1]: values})
    return cw_cl_counts # Returning words and tag with "_" in between them.
###############################################################################################################################################################
##It is providing us the list of tags.
###############################################################################################################################################################

def word_tag_seperator(sent): # Creating a seperate list of tag in this function from evry sentence.
    tag_of_word= []
    for tuple in sent:
        word,tag = tuple
        tag_of_word.append(tag)
    tag_of_word=tag_of_word
    return tag_of_word #list of tags.
###############################################################################################################################################################
## It is providing us with word and tag count per sentence.
###############################################################################################################################################################
def Phi(sentence, phi1count): # giving sentence and cw_cl_counts as input to calculate frequenncy per sentence.
    c = Counter(sentence)
    d = {}
    for word, count in c.items():
        underscore = word[0] + '_' + word[1] #joining words and tags with "_".
        if underscore not in phi1count:
            d.update({underscore : 0})
        else:
            d.update({underscore : count}) #updating count.
    return d
#################################################################################################################################################################
##Training function providing us with the updated weights using structured binary perceptron .
###############################################################################################################################################################
def train (data, counts): #training dataset working on input "cw_cl_counts" and "train data"
    weights= {}
    for key, values in counts.items():
        weights[key] = 0
    for i in range(2):
        Possible_combination_tag=[]
        random.shuffle(data) #shuffling training data
        for j in range(1,6): # we have 5 tag so, using 5 as range in for loop.
            tags = itertools.combinations_with_replacement(tag_list, j)
            Possible_combination_tag.append(tags)
        for sentence in data:
            new_Possible_combination_var=[p for p in itertools.product(tag_list, repeat=len(sentence))]
            new_Possible_combination_var=[list(mqm) for mqm in new_Possible_combination_var] #generating possibile tags.
            word_in_sentence = [x[0] for x in sentence]
            scores = []
            for k in new_Possible_combination_var:
                tuple = list(zip(word_in_sentence, k))
                call_Phi = Phi(tuple,counts) #calling function Phi.
                score = 0
                for key, value in call_Phi.items(): #Updating weights.
                    if key in weights:
                        score = score + (value* weights[key])
                    else:
                        continue
                scores.append(score)
            Max_value = max(scores)
            index_max = scores.index(Max_value)
            Y_hat = list(new_Possible_combination_var[index_max]) # taking input for Y_hat from combination of tags.
            Y = [x[1] for x in sentence] #taking input for Y from sentence.
            if Y_hat != Y: #comparing both Y and Y_hat.
                tuple_1 = list(zip(word_in_sentence, Y))
                tuple_2 = list(zip(word_in_sentence, Y_hat))
                Real_Phi = Phi(tuple_1,counts)  # calculating phi using "Y".
                predict_Phi = Phi(tuple_2,counts) # calculating phi using "Y_hat".
                for key in Real_Phi:
                    if key in weights.keys():
                        weights[key] += Real_Phi[key]
                for key in predict_Phi:
                    if key in weights.keys():
                        weights[key] -= predict_Phi[key]
    return weights #Returning weights.
###############################################################################################################################################################
##Test function providing us with the f1score on the basis of weights,Y and Y_hats.
###############################################################################################################################################################
def test(data, counts, test_weights): #test function having input test data, counts and test weights.
    Possible_combination_tag=[]
    for i in range(1,6):
        tags = itertools.combinations_with_replacement(tag_list, i)
        Possible_combination_tag.append(tags)
    Y_predict = []
    Y_correct = []
    for i in data:
        new_Possible_combination_var = [p for p in itertools.product(tag_list, repeat=len(i))] #possible tags.
        new_Possible_combination_var = [list(mqm) for mqm in new_Possible_combination_var] #
        word_in_sentence = [x[0] for x in i] #taking words from sentence.
        scores = []
        for j in new_Possible_combination_var:
            score =0
            tuple = list(zip(word_in_sentence, j))
            call_Phi = Phi(tuple, counts) #calling function Phi.
            for key, value in call_Phi.items():
                if key in test_weights:
                    score = score + (value * test_weights[key])
                else:
                    continue
            scores.append(score)
        Max_value = max(scores)
        index_max = scores.index(Max_value)
        Y_hat = list(new_Possible_combination_var[index_max]) # calculating phi using "Y_hat".
        Y_predict.append(Y_hat)
        tag = word_tag_seperator(i)
        Y_correct.append(tag) #calculating phi using "Y"
    Y_correct_list = [item for sublist in Y_correct for item in sublist] # multiply list into one list.
    Y_predict_list = [item for sublist in Y_predict for item in sublist] # multiply list into one list.
    f1_micro = f1_score(Y_correct_list, Y_predict_list, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC']) #f1 score.
    print("f1score for Phi1",f1_micro)
###############################################################################################################################################################
##Extracting Top 10 features.
###############################################################################################################################################################
def top10(data): # top 10 features.
    ten_top = [] # top 10 features list.
    for i in tag_list: # running iteration over tag_list declared above globally.
        ten_top_feature_dict = {} # dictionary for top 10 feature.
        for key, value in data.items():
            if key.endswith("_" + i):
                ten_top_feature_dict.update({key:value})
        ten_top.append(ten_top_feature_dict) # top 10 tag.
    for d in ten_top:
        sorted_val = sorted(d.items(),key = lambda key: key[1], reverse = True)[:10]
        top10_features = list(d.keys())[0].split("_")[1] #top 10 words.
        print("Top 10 features of " + top10_features + " are\n",sorted_val) #printing the top 10 feature.

###############################################################################################################################################################
##It is providing us with tag and tag count
###############################################################################################################################################################
def Extraction_of_feature_2(data):  # extracting tags for the comparison in the training model.
    merged_list = list(itertools.chain.from_iterable(data))
    tag_Freq = [x[1] for x in merged_list]
    tag_Freq_iter = iter(tag_Freq)
    tags_Phi2= [i + "_" +next(tag_Freq_iter, ' ') for i in tag_Freq_iter]
    tags_Phi2_count =Counter(tags_Phi2)
    return tags_Phi2_count
###############################################################################################################################################################
## It is providing us with word-tag count and tag- tag count per sentence.
###############################################################################################################################################################
def Phi_2(sentence, phi2count, merger):
    new_sentence = []
    N_N_sentence = [x[0] for x in sentence]
    new_sentence.append(["None"] + N_N_sentence)
    N_N_sentence=[x[1] for x in sentence]
    new_sentence.append(["None"]+N_N_sentence)
    bi = n_grams_generation(new_sentence[1], 2)
    countobj =Counter(bi)
    bigramdict = {}
    for k, v in countobj.items():
        k = k.replace(' ','_')
        if k in phi2count:
            bigramdict.update({k : v})
        else:
            bigramdict.update({k : 0})
    bigramdict = combined_dictionary(bigramdict, merger)
    return bigramdict
###############################################################################################################################################################
##Bigrams for Tags.
###############################################################################################################################################################
def n_grams_generation(tokens, n_gram):
    ngrams = zip(*[tokens[new_sentence:] for new_sentence in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]
###############################################################################################################################################################
##This function gives dictionary of the combination of Phi and Phi_2 function.
###############################################################################################################################################################
def combined_dictionary(*dict_args):
    combined_dictionary_dict={}
    for dictionary in dict_args:
        combined_dictionary_dict.update(dictionary)
    return combined_dictionary_dict
#################################################################################################################################################################
##Training function providing us with the averge weights using structured binary perceptron taking merged dictionary and training data as input.
###############################################################################################################################################################
def train_2 (data, phi1count, phi2count): #training dataset working on input "cw_cl_counts" and "train data"
    weights= {}
    averge_of_weights=[]
    for key, values in phi2count.items():
        weights[key] = 0
    for key, values in phi1count.items():
        weights[key] = 0
    for i in range(10):
        Possible_combination_tag=[]
        random.shuffle(data) #shuffling training data
        for j in range(1,6): # we have 5 tag so, using 5 as range in for loop.
            tags = itertools.combinations_with_replacement(tag_list, j)
            Possible_combination_tag.append(tags)
        for sentence in data:
            new_Possible_combination_var=[p for p in itertools.product(tag_list, repeat=len(sentence))]
            word_in_sentence = [x[0] for x in sentence]
            scores = []
            #list_merge = []
            for k in new_Possible_combination_var:
                tuple = list(zip(word_in_sentence, k))
                phi1 = Phi(tuple, phi1count)
                call_Phi = Phi_2(tuple,phi2count,phi1) #calling function Phi2
                score = 0
                for key, value in call_Phi.items(): #Updating weights.
                    if key in weights:
                        score = score + (value* weights[key])
                    else:
                        continue
                scores.append(score)
            Max_value = max(scores)
            index_max = scores.index(Max_value)
            Y_hat = list(new_Possible_combination_var[index_max]) # taking input for Y_hat from combination of tags.
            Y = [x[1] for x in sentence] #taking input for Y from sentence.
            if Y_hat != Y: #comparing both Y and Y_hat.
                tuple_1 = list(zip(word_in_sentence, Y))
                tuple_2 = list(zip(word_in_sentence, Y_hat))
                Real_Phi = Phi(tuple_1,phi1count)  # calculating phi using "Y".
                Real_Phi=Phi_2(tuple_1, phi2count, Real_Phi)
                predict_Phi=Phi(tuple_2, phi1count)
                predict_Phi = Phi_2(tuple_2,phi2count, predict_Phi) # calculating phi using "Y_hat".
                for key in Real_Phi:
                    if key in weights.keys():
                        weights[key] += Real_Phi[key]
                for key in predict_Phi:
                    if key in weights.keys():
                        weights[key] -= predict_Phi[key]
        averge_of_weights.append(copy.deepcopy(weights))
        summer = Counter()
        counter = Counter()
        for i in averge_of_weights:
            summer.update(i)
            counter.update(i.keys())
        avg = {x: float(summer[x])/counter[x] for x in summer.keys()}
    return avg #Returning averge weights.
#################################################################################################################################################################
##Test function providing us with the f1score on the basis of test data,Phi ,Phi_2 and training data.
###############################################################################################################################################################
def test_2(data, phi1count, phi2count, test_weights): #test function having input test data, counts and test weights.
    Possible_combination_tag=[]
    for i in range(1,6):
        tags = itertools.combinations_with_replacement(tag_list, i)
        Possible_combination_tag.append(tags)
    Y_predict = [] #list for Y predict.
    Y_correct = [] #list for Y correct.
    for i in data:
        new_Possible_combination_var = [p for p in itertools.product(tag_list, repeat=len(i))] #possible tags.
        new_Possible_combination_var = [list(mqm) for mqm in new_Possible_combination_var] #
        word_in_sentence = [x[0] for x in i] #taking words from sentence.
        scores = []
        for j in new_Possible_combination_var:
            score =0
            tuple = list(zip(word_in_sentence, j))
            Phi1= Phi(tuple, phi1count) #calling function Phi.
            call_Phi = Phi_2(tuple, phi2count, Phi1)

            for key, value in call_Phi.items():
                if key in test_weights:
                    score = score + (value * test_weights[key])
                else:
                    continue
            scores.append(score)
        Max_value = max(scores)
        index_max = scores.index(Max_value)
        Y_hat = list(new_Possible_combination_var[index_max]) # calculating phi using "Y_hat".
        Y_predict.append(Y_hat)
        tag = word_tag_seperator(i)
        Y_correct.append(tag) #calculating phi using "Y"
    Y_correct_list = [item for sublist in Y_correct for item in sublist] # multiply list into one list.
    Y_predict_list = [item for sublist in Y_predict for item in sublist] # multiply list into one list.
    f1_micro = f1_score(Y_correct_list, Y_predict_list, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC']) #f1 score.
    print("f1score for Phi1 and Phi2",f1_micro)
###############################################################################################################################################################
##Extracting Top 10 features.
###############################################################################################################################################################
def top_10(data): # top 10 features.
    ten_top = [] # top 10 features list.
    for i in tag_list: # running iteration over tag_list declared above globally.
        ten_top_feature_dict = {} # dictionary for top 10 feature.
        for key, value in data.items():
            if key.endswith("_" + i):
                ten_top_feature_dict.update({key:value})
        ten_top.append(ten_top_feature_dict) # top 10 tag.
    for d in ten_top:
        sorted_val = sorted(d.items(),key = lambda key: key[1], reverse = True)[:10]
        top10_features = list(d.keys())[0].split("_")[1] #top 10 words.
        print("Top 10 features of " + top10_features + " are\n",sorted_val) #printing the top 10 feature.

##################################################################Calling Function##############################################################################################
###############################################################################################################################################################
## Calling function for Phi1
###############################################################################################################################################################
train_data = load_dataset_sents(train_file)
E_o_f = Extraction_of_feature(train_data)
tr=train(train_data, E_o_f)
test_data = load_dataset_sents(test_file)
ccc= test(test_data,E_o_f,tr)
top10(tr)
###############################################################################################################################################################
## Calling function for Phi2
###############################################################################################################################################################
E_o_f2=Extraction_of_feature_2(train_data)
tr_2=train_2(train_data,E_o_f,E_o_f2)
test_data = load_dataset_sents(test_file)
ccc= test_2(test_data,E_o_f,E_o_f2,tr_2)
top_10(tr_2)