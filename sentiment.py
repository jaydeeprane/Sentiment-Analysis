import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model 
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec=feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.

    
    # Creating dictionaries of positive and negative words occuring in reviews/tweets
    posFinalDict={}
    for sentence in train_pos:
        for word in set(sentence):
            if word in posFinalDict:
                posFinalDict[word]+=1
            else:
                posFinalDict[word]=1

    negFinalDict={}
    for sentence in train_neg:
        for word in set(sentence):
                if word in negFinalDict:
                    negFinalDict[word]+=1
                else:
                    negFinalDict[word]=1


    # Eliminating the stopwords from the Dictionaries
    posWSw = { k:v for k, v in posFinalDict.items() if v not in stopwords }
    negWSw = { k:v for k, v in negFinalDict.items() if v not in stopwords }
    
    # Satisfying the 1% condition given
    posFinalDict1 = { k:v for k, v in posWSw.items() if v>=(0.01*len(train_pos)) }
    negFinalDict1 = { k:v for k, v in negWSw.items() if v>=(0.01*len(train_neg)) }
    
    featureList = {}

    # Satisfying the 2% condition and forming final list of feature words
    for word in posFinalDict1.keys():
        if ((not word in negFinalDict) or (posFinalDict1[word]>=2*negFinalDict[word])):
            featureList[word]=1
    
    for word in negFinalDict1.keys():
        if ((not word in posFinalDict) or (negFinalDict1[word]>=2*posFinalDict[word])):
            featureList[word]=1
    
    # print "Length of Feature list:",len(featureList)
    # sys.exit(0)

    

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
 
    # Updating the vectors to be returned based on feature list generated
    featureList=featureList.keys()
    
    train_pos_vec=[]

    for sentence in train_pos:
        sentenceDict ={}
        for word in sentence:
            sentenceDict[word] = 1
        binaryList=[]
        for word in featureList:
            if word in sentenceDict:
                binaryList.append(1)
            else:
                binaryList.append(0)
        train_pos_vec.append(binaryList)
    train_neg_vec=[]

    for sentence in train_neg:
    	sentenceDict = {}
        for word in sentence:
        	sentenceDict[word] = 1
        binaryList=[]
        for word in featureList:
         if word in sentenceDict:
                binaryList.append(1)
         else:
                binaryList.append(0)
        train_neg_vec.append(binaryList)
    
    test_pos_vec=[]

    for sentence in test_pos:
    	sentenceDict = {}
        for word in sentence:
         sentenceDict[word] = 1	
        binaryList=[]
        for word in featureList:
         if word in sentenceDict:
                binaryList.append(1)
         else:
                binaryList.append(0)
        test_pos_vec.append(binaryList)

    test_neg_vec=[]

    for sentence in test_neg:
    	sentenceDict = {}
        for word in sentence:
        	sentenceDict[word] = 1	
        binaryList=[]
        for word in featureList:
         if word in sentenceDict:
                binaryList.append(1)
         else:
                binaryList.append(0)
        test_neg_vec.append(binaryList)
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
   

    # Creating and updating labelled sentence objects
    labeled_train_pos=[]
    
    for i,reviewWords in enumerate(train_pos):
        lso=LabeledSentence(words=reviewWords,tags=['TRAIN_POS_'+str(i)])
        labeled_train_pos.append(lso)

    labeled_train_neg=[]
    for i,reviewWords in enumerate(train_neg):
        lso=LabeledSentence(words=reviewWords,tags=['TRAIN_NEG_'+str(i)])
        labeled_train_neg.append(lso)

    
    labeled_test_pos=[]
    for i,reviewWords in enumerate(test_pos):
        lso=LabeledSentence(words=reviewWords,tags=['TEST_POS_'+str(i)])
        labeled_test_pos.append(lso)

    
    labeled_test_neg=[]
    for i,reviewWords in enumerate(test_neg):
        lso=LabeledSentence(words=reviewWords,tags=['TEST_NEG_'+str(i)])
        labeled_test_neg.append(lso)
    
	
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)
    
    # Use the docvecs function to extract the feature vectors for the training and test data
  
    train_pos_vec=[]

    for i,fv in enumerate(train_pos):
    	featureVec = model.docvecs['TRAIN_POS_'+str(i)] 
        train_pos_vec.append(featureVec)
   
    train_neg_vec=[]

    for i,fv in enumerate(train_neg):
    	featureVec = model.docvecs['TRAIN_NEG_'+str(i)]
        train_neg_vec.append(featureVec)
    
    test_pos_vec=[]

    for i,fv in enumerate(test_pos):
    	featureVec = model.docvecs['TEST_POS_'+str(i)]
        test_pos_vec.append(featureVec)
    
    test_neg_vec=[]

    for i,fv in enumerate(test_neg):
	  	test_neg_vec.append(model.docvecs['TEST_NEG_'+str(i)])

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
   
    X = train_pos_vec + train_neg_vec
    bm = sklearn.naive_bayes.BernoulliNB(alpha=1.0,binarize=None)
    nb_model = bm.fit(X, Y)

    
    lm = sklearn.linear_model.LogisticRegression()
    lr_model = lm.fit(X,Y)

    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
   
    X = train_pos_vec + train_neg_vec

    gm = sklearn.naive_bayes.GaussianNB()
    nb_model = gm.fit(X,Y)

    lm = sklearn.linear_model.LogisticRegression()
    lr_model = lm.fit(X,Y)
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
 
    predictPos = list(model.predict(test_pos_vec))
    predictNeg = list(model.predict(test_neg_vec))

    tp = predictPos.count('pos')
    fn = predictPos.count('neg')
    tn = predictNeg.count('neg')
    fp = predictNeg.count('pos')
    accuracy = float((tp+tn))/(len(test_pos_vec)+len(test_neg_vec))
    # print"",predictPos
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()  
