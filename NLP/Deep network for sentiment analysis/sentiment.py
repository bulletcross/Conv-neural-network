import nltk
#For creating a dictionary of parsed words
from nltk.tokenize import word_tokenize
#For sanatize words with verd, tenses into a single word
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
#For quick access of data
import pickle
#for evicting words which are common
from collections import Counter

#This will remove ing from eating to make eat
lemmatizer = WordNetLemmatizer()
#Defining max lines we would like to read, memory constraint
n_line = 10000000

#Create a sanatized dictionary out of two files
def create_lexicon(pos,neg):
    lexicon = []
    for f in [pos,neg]:
        with open(f,'r') as open_file:
            contents = open_file.readlines()
            for line in contents[:n_line]:
                #Ascii code problem with this version of nltk
                try:
                    all_words = word_tokenize(line.lower())
                except ValueError:
                    print(line)
                lexicon = lexicon + list(all_words)
    print(len(lexicon))
    lexicon = [lemmatizer.lemmatize(i.decode('utf-8')) for i in lexicon]
    #Counts the appearance of words in both files, in form of dictionary
    w_counts = Counter(lexicon)
    #Final lexicon will have only important words we would need
    final_lexicon = []
    for w in w_counts:
        if w_counts[w] < 1000 and w_counts[w] > 50:
            final_lexicon.append(w)
    print(len(final_lexicon))
    return final_lexicon

"""For a given sample, lexicon and classification, it will create
create a features set of form [[[0 1 0 0 1 ..], [feature]],[],[],[],....]
each element in list represent a tuple of encoding of lexicon supplied and
corresponding feature(positive or negative). Each element corresponds
to a line supplied in sample ecoded for given lexicon"""
def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample,'r') as open_file:
        contents = open_file.readlines()
        for line in contents[:n_line]:
            try:
                current_words = word_tokenize(line.lower())
                current_words = [lemmatizer.lemmatize(i.decode('utf-8')) for i in current_words]
                features = np.zeros(len(lexicon))
                for word in current_words:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] = features[index_value] + 1
                features  = list(features)
                featureset.append([features, classification])
            except ValueError:
                print(line)
    return featureset

#By default test size is 10%
def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features = features + sample_handling('pos.txt',lexicon,[1,0])
    features = features + sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)

    features = np.array(features)
    testing_size = int(test_size*len(features))
    #Seperate test data
    train_x = list(features[:,0][:-testing_size])#Takes all one hot arrays in list
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y

if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
    with open('sentiment_set.pickle','wb') as open_file:
        pickle.dump([train_x,train_y,test_x,test_y],open_file)
