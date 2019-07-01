from collections import Counter
import sys
import itertools
import numpy as np
import time, random
from sklearn.metrics import f1_score



random.seed(11242)
depochs = 5
feat_red = 0

print("\nDefault no. of epochs: ", depochs)
print("\nDefault feature reduction threshold: ", feat_red)

print("\nLoading the data \n")

"""Loading the data"""


### Load the dataset
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
    return zip_inps if as_zip else (inputs, targets)


train_data = load_dataset_sents(sys.argv[2])
test_data = load_dataset_sents(sys.argv[3])

## unique tags
all_tags = ["O", "PER", "LOC", "ORG", "MISC"]

""" Defining our feature space """

print("\nDefining the feature space \n")


# feature space of cw_ct
def cw_ct_counts(data, freq_thresh=5):  # data inputted as (cur_word, cur_tag)

    cw_c1_c = Counter()

    for doc in data:
        cw_c1_c.update(Counter(doc))

    return Counter({k: v for k, v in cw_c1_c.items() if v > freq_thresh})


cw_ct_count = cw_ct_counts(train_data, freq_thresh=feat_red)


# feature representation of a sentence cw-ct
def phi_1(sent, cw_ct_count):  # sent as (cur_word, cur_tag)
    for i in sent:
        if i in cw_ct_count.keys():
            count = 1
        else:
            count = 0
    return count


def scoring(doc, weights, algoithum = 1): # scoring function call the both the beam and viterbi function by giving them sentence ,tag ,weight and counts in the argument.
    # unzippin them
    sentence, tags = list(zip(*doc))
    if algoithum==1:
        genrate_sequence = viterbi(sentence,all_tags,weights,cw_ct_count)
    else:
        genrate_sequence = beam_search(sentence, all_tags, weights, cw_ct_count, 5)
    zip_generate_sequence = [(word,tag) for word ,tag in zip(sentence,genrate_sequence)]
    return zip_generate_sequence


def viterbi(sentence, all_tag,weights,cw_ct_count):# giving sentence ,tag , weight and count in the argument.
    viterbi_matrix = np.zeros((len(all_tag),len(sentence))) #making matrix with zero values.
    for i in range(len(all_tags)): # finding the probabilities of word and tag combination in column 1.
        list_of_tuple = [(sentence[0],all_tags[i])]
        calling_phi1 = phi_1(list_of_tuple,cw_ct_count) #calling function.
        viterbi_matrix[i][0] = calling_phi1 * weights[(sentence[0], all_tags[i])]# finding best probability of word-tag combination.
    for j in range(1,len(sentence)):
        for k in range(len(all_tag)): # finding the probabilities of word and tag combination in column other than column1.
            list_of_tuple = [(sentence[j], all_tags[k])]
            calling_phi1 = phi_1(list_of_tuple, cw_ct_count) #calling function.
            maximum_val_col = np.amax(viterbi_matrix[:,j-1]) #calculating maximum values of word-tag combination for the previous column.
            viterbi_matrix[k][j] = maximum_val_col + calling_phi1 * weights[(sentence[j], all_tags[k])] #finding best probability of word-tag combination.
    index_of_maxval = np.argmax(viterbi_matrix,axis = 0)
    sequence_of_tag = [all_tags[x] for x in index_of_maxval]
    return sequence_of_tag
import heapq
def beam_search(sentence, all_tag,weights,cw_ct_count,b_s):# giving sentence ,tag , weight and count in the argument.

    beam_matrix = np.zeros((len(all_tag), len(sentence))) #making matrix with zero values.
    for i in range(len(all_tags)): # finding the probabilities of word and tag combination in column 1.
        list_of_tuple = [(sentence[0], all_tags[i])]
        calling_phi1 = phi_1(list_of_tuple, cw_ct_count) #calling function.
        beam_matrix[i][0] = calling_phi1 * weights[(sentence[0], all_tags[i])] # finding best probability of word-tag combination.
    for j in range(1, len(sentence)): # finding the probabilities of word and tag combination in column other than column1.
        top_k_indices = heapq.nlargest(b_s, range(len(beam_matrix[j-1])), beam_matrix[j-1].take)
        for k in range(len(all_tag)):
            list_of_tuple = [(sentence[j], all_tags[k])]
            calling_phi1 = phi_1(list_of_tuple, cw_ct_count)#calling function.
            maximum_val_col = -300
            for a in top_k_indices: #finding maximum previous value.
                if maximum_val_col< beam_matrix[a,j-1]:
                    maximum_val_col = beam_matrix[a,j-1] #calculating maximum values of word-tag combination for the previous column.
            beam_matrix[k][j] = maximum_val_col + calling_phi1 * weights[(sentence[j], all_tags[k])] #finding best probability of word-tag combination.
    index_of_maxval = np.argmax(beam_matrix, axis=0)
    sequence_of_tag = [all_tags[x] for x in index_of_maxval]
    return sequence_of_tag


def train_perceptron( data, epochs, shuffle=True,algoithum = 1):
    # variables used as metrics for performance and accuracy
    iterations = range(len(data) * epochs)
    false_prediction = 0
    false_predictions = []

    # initialising our weights dictionary as a counter
    # counter.update allows addition of relevant values for keys
    # a normal dictionary replaces the key-value pair
    weights = Counter()

    start = time.time()

    # multiple passes
    for epoch in range(epochs):
        false = 0
        now = time.time()

        # going through each sentence-tag_seq pair in training_data

        # shuffling if necessary
        if shuffle == True:
            random.shuffle(data)

        for doc in data:

            # retrieve the highest scoring sequence
            max_scoring_seq = scoring(doc, weights,algoithum=algoithum)

            # if the prediction is wrong
            if max_scoring_seq != doc:
                correct = Counter(doc)

                # negate the sign of predicted wrong
                predicted = Counter({k: -v for k, v in Counter(max_scoring_seq).items()})

                # add correct
                weights.update(correct)

                # negate false
                weights.update(predicted)

                """Recording false predictions"""
                false += 1
                false_prediction += 1
            false_predictions.append(false_prediction)

        print("Epoch: ", epoch + 1,
              " / Time for epoch: ", round(time.time() - now, 2),
              " / No. of false predictions: ", false)

    return weights, false_predictions, iterations


# testing the learned weights
def test_perceptron( data, weights,algoithum = 1):
    correct_tags = []
    predicted_tags = []

    i = 0

    for doc in data:
        _, tags = list(zip(*doc))

        correct_tags.extend(tags)

        max_scoring_seq = scoring(doc, weights,algoithum=algoithum)

        _, pred_tags = list(zip(*max_scoring_seq))

        predicted_tags.extend(pred_tags)

    return correct_tags, predicted_tags


def evaluate( correct_tags, predicted_tags):
    f1 = f1_score(correct_tags, predicted_tags, average='micro', labels=["PER", "LOC", "ORG", "MISC"])

    print("F1 Score: ", round(f1, 5))

    return f1


#Calling functions.


# calling viterbi function.
if sys.argv[1]=='-v':

    print("\n Viterbi")

    print("\nTraining the perceptron with (cur_word, cur_tag) \n")

    weights, false_predictions, iterations = train_perceptron(train_data, epochs=depochs,algoithum =1)

    print("\nEvaluating the perceptron with (cur_word, cur_tag) \n")

    correct_tags, predicted_tags = test_perceptron(test_data, weights,algoithum = 1)

    f1 = evaluate(correct_tags, predicted_tags)



# calling beam search function.

elif  sys.argv[1]=='-b':

    print("\n beam search")

    print("\nTraining the perceptron with (cur_word, cur_tag) \n")

    weights, false_predictions, iterations = train_perceptron(train_data, epochs=depochs,algoithum =2)

    print("\nEvaluating the perceptron with (cur_word, cur_tag) \n")

    correct_tags, predicted_tags = test_perceptron(test_data, weights,algoithum =2)

    f1 = evaluate(correct_tags, predicted_tags)


