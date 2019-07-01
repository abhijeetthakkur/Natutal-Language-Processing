###########################################Importing Libraries###########################################
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
##########################################Variable Decleartion############################################
counter_sanity_check = 0 #counter used in the sanity check code
counter_prediction = 0 #counter used in the prediction check code
answer={} #result from the prediction code.
score_list={} 
torch.manual_seed(1)
CONTEXT_SIZE = 2
upgraded_data = ""
EMBEDDING_DIM = 10
list_of_sent = [] 
#########################################sentences#######################################################
test_sentences = "The mathematician ran . ; The mathematician ran to the store . ; The physicist ran to the store . ; " \
              "The philosopher thought about it . ; The mathematician solved the open problem ."
test_sentence = "The ______ solved the open problem ."
word_to_be_predicted = ["physicist", "philosopher", "mathematician"]
#######################################tokenisation of sentence#########################################
all_test_sentences = test_sentences.split(";")
for i in all_test_sentences:
    upgraded_all_test_sentences = "start " + i + " stop "
    upgraded_data = upgraded_data + upgraded_all_test_sentences
    list_of_sent.append(upgraded_all_test_sentences.split())
vocab = set(upgraded_data.split())
word_to_ix = {word: i for i, word in enumerate(vocab)}
word_to_ix_index = {i: w for w, i in word_to_ix.items()}

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs, self.embeddings
    
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(15):
    total_loss = torch.Tensor([0])
    for sentence in list_of_sent:
        trigrams = [([sentence[i], sentence[i + 1]], sentence[i + 2])
                for i in range(len(sentence) - 2)]
        for context, target in trigrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in variables)
            context_idxs = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_idxs))

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs, embeddings = model(context_var)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a variable)
            loss = loss_function(log_probs, autograd.Variable(
                torch.LongTensor([word_to_ix[target]])))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            total_loss += loss.data
    losses.append(total_loss)
#print(losses)  # The loss decreased every iteration over the training data!
############################################Sanity Check###########################################
#selecting the stence from list.
sanity_check = list_of_sent[1]
# generating trigrams of above selected sentence.
trigrams_sanity_check = [([sanity_check[i], sanity_check[i + 1]], sanity_check[i + 2])
            for i in range(len(sanity_check) - 2)]
#Using above code
for previous_words, pred in trigrams:
     context_idxs = [word_to_ix[w] for w in previous_words]
     context_var = autograd.Variable(torch.LongTensor(context_idxs))
     model.zero_grad()
     log_probs, embeddings = model(context_var)
     max_value = max(log_probs[0])
     max_val_list = list(log_probs[0])
     max_index = max_val_list.index(max_value)
     tag_word = word_to_ix_index[max_index]
     print((context, tag_word))
     if pred == tag_word: #to check the prediction score.
         counter_sanity_check = counter_sanity_check + 1
     else:
         counter_sanity_check = counter_sanity_check + 0
prediction = format(counter_sanity_check/len(trigrams_sanity_check))
print("Prediction",prediction)
###################################Prediction#################################################
for j in word_to_be_predicted: #for loop on the set of 3 words given for the prediction.
    sent = test_sentence.replace("______",j) # replaceing blank space with the predicted word.
    sent_upgrade = sent.split() 
    sent_upgrade_trigram = [([sent_upgrade[i], sent_upgrade[i + 1]], sent_upgrade[i + 2])
                                for i in range(len(sent_upgrade) - 2)] #geneating trigrams.
#using above code.
    for context, true in sent_upgrade_trigram:
         context_idxs = [word_to_ix[w] for w in context]
         context_var = autograd.Variable(torch.LongTensor(context_idxs))
         model.zero_grad()
         log_probs, embeddings = model(context_var)
         log_probs_list = list(log_probs[0])
         summation_log_probs_list = sum(log_probs_list)
         counter_prediction = counter_prediction + summation_log_probs_list
    ll_tensor = torch.LongTensor([word_to_ix[j]]) #taking code from doc given.
#to find the predicted word, doing indexing and choosing word from the dictionary.    
    score = embeddings(autograd.Variable(ll_tensor))
    answer[j] = counter_prediction   
    score_list[j] = score
score_index = {score: w for w, score in answer.items()}
score_index_max = max(score_index.keys())
word = score_index[score_index_max]
print(word)    
print(score_index_max)
####################################CosineSimilarity##############################################
math = score_list["mathematician"]
physicist = score_list["physicist"]
philosopher = score_list["philosopher"]
cos = nn.CosineSimilarity(dim=1, eps=1e-6) #using formula.
output = cos(physicist,math)
output1 = cos(philosopher,math)
print("Cosine similarity of physicist and mathematician",output)
print("Cosine similarity of philosopher and mathematician",output1)
####################################END OF CODE################################################
     
    


         
     
     
