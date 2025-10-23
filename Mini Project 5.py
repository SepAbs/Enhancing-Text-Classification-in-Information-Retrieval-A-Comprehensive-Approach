import nltk, itertools
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import FastText, KeyedVectors, Word2Vec
from gensim.models.phrases import Phraser, Phrases
# from glove import *
from Levenshtein import distance
from matplotlib.pyplot import annotate, box, plot, savefig, scatter, show, subplots, title, xlabel, ylabel
from matplotlib.colors import ListedColormap
from numpy import add, array, transpose, zeros
from os import listdir
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from string import punctuation

def preProcessor(String):
    stopWords, String, Specials = stopwords.words('english'), wordpunct_tokenize(String), [").", "\'", ",", ";", ".<", "br", "/><", "/>"] # Case Folding & Tokenisations
    return [Token.lower() for Token in String if Token not in punctuation and Token.lower() not in stopWords and Token not in Specials]

def Approx(unknownWord, Vocabulary):
    # Listing similar words in the vocabulary to the unknown words.
    listCounter = [distance(unknownWord, Word) for Word in Vocabulary]

    # Retruning most similar word exists.
    return Vocabulary[listCounter.index(min(listCounter))]

def term_doc_mat_LSA():
    Topics, Title = {0: "Like", 1: "Disike"}, "SVD dropoff"
    LSA = LSI.fit_transform(tfidf.fit_transform(X_train))
    print(f"\nSigma:\n{LSI.singular_values_}\nV^T:\n{LSI.components_.T}") # Sigmas & VT     
    Terms = tfidf.get_feature_names_out()
    for Index, Component in enumerate(LSI.components_):
        print(f"{Topics[Index]}: {list(dict(sorted(zip(Terms, Component), key = lambda t: t[1], reverse = True)[:5]).keys())}\n")

    print(LSI.explained_variance_ratio_)
    plot(range(2), LSI.explained_variance_ratio_ * 100)
    xlabel("Topic Number")
    ylabel("% explained")
    title(Title)
    savefig(Title, dpi = 1200)
    show()  # show first chart

def documentLSA():
    tfX_train, tfidfX_train, tfX_test = tfVectorizer.fit_transform(X_train), tfidfVectorizer.fit_transform(X_train), tfVectorizer.transform(X_test)
    tf_doc_term_matrix, tfidf_doc_term_matrix = tfX_train.toarray(), tfidfX_train.toarray() # Ordered document-term by tf.
    sortedWords = sorted(list(set([Word for Sentence in Sentences for Word in Sentence])))[:tf_doc_term_matrix.shape[1]] # Dimension reduction for creating document vector matrix. (A logical error occures that latter words are not considered.)
    term_doc_matrix, wordEmbeddings, validResponses = {"TF": "tf term-document matrix", "IDF": "tf-idf term-document matrix"}, {"W2V": "Word2Vec", "GLV": "GloVe", "FT": "FastText"}, ["TF", "IDF"]
    while True:
        term_doc_matrix_form = input("Tf term-document matrix or tf-idf term-document matrix? <Tf, Idf> ").upper()
        if term_doc_matrix_form in validResponses:
            print(f"\nYou chose {term_doc_matrix[term_doc_matrix_form]} matrix form.")
            break
        print("\nSelect one, dude!")

    validResponses = ["W2V", "GLV", "FT"]
    while True:
        word_embedding_technique = input("Word2Vec, GloVe or FastText? <W2V, GLV, FT> ").upper()
        if word_embedding_technique in validResponses:
            print(f"\nYou chose {wordEmbeddings[word_embedding_technique]} word embedding technique.")
            break
        print("\nSelect one, dude!")

    if term_doc_matrix_form == "TF":
        # transpose(document-term matrix * word-embedded word vector matrix) = embedded word vector-document matrix.
        if word_embedding_technique == "W2V": 
            return SVD.fit_transform(csr_matrix(transpose(tf_doc_term_matrix.dot(WordVec(Sentences)))))

        elif word_embedding_technique == "GLV":
            return SVD.fit_transform(csr_matrix(transpose(tf_doc_term_matrix.dot(GloVe(Sentences)))))
            
        return SVD.fit_transform(csr_matrix(transpose(tf_doc_term_matrix.dot(fastText(Sentences)))))

    if word_embedding_technique == "W2V":
        return SVD.fit_transform(csr_matrix(transpose(tfidf_doc_term_matrix.dot(WordVec(Sentences)))))

    elif word_embedding_technique == "GLV":
        return SVD.fit_transform(csr_matrix(transpose(tfidf_doc_term_matrix.dot(GloVe(Sentences)))))
            
    return SVD.fit_transform(csr_matrix(transpose(tfidf_doc_term_matrix.dot(fastText(Sentences)))))

def DocVec():
    taggedData = [TaggedDocument(words = trainDoc.split(), tags = [str(Tag)]) for Tag, trainDoc in enumerate(X_train)]
    # train the Doc2vec model
    Model = Doc2Vec(vector_size = 300, min_count = 2, epochs = 50)
    Model.build_vocab(taggedData)
    Model.train(taggedData, total_examples = Model.corpus_count, epochs = Model.epochs)
    return [Model.infer_vector(trainDoc.split()) for trainDoc in X_train]

def WordVec(Sentences):
    validResponses = ["CBOW", "SG"]
    while True:
        modelMethod = input("CBOW or Skip-gram? <CBOW, Sg> ").upper()
        if modelMethod in validResponses:
            print("Alright!")
            break
        print("Select one, dude!")

    if modelMethod == "CBOW":
        # Create CBOW model
        Model = Word2Vec(min_count = 5, vector_size = 300, window = 5)
    else:
        # Create Skip-gram model(sg = 1)
        Model = Word2Vec(min_count = 5, sg = 1, vector_size = 300, window = 5)

    Model.build_vocab(Sentences, progress_per = 1000)

    # train on our data
    Model.train(Sentences, epochs = 100, total_examples = len(Sentences))

    return Model.wv
    
def GloVe(Sentences):
    # Stanford pretrained
    # Load GloVe embeddings using Gensim
    gloVectors = KeyedVectors.load_word2vec_format("glove.6B.100d.txt", binary = False, no_header = True) # New version!

    # build a toy model to update with
    Model = Word2Vec(min_count = 5, vector_size = 300)
    Model.build_vocab(Sentences)
    numberExamples = Model.corpus_count

    # add GloVe's vocabulary & weights.
    Model.build_vocab([list(gloVectors.index_to_key)], update = True)

    # train on our data
    Model.train(Sentences, epochs = Model.epochs, total_examples = numberExamples)

    return Model.wv

def fastText(Sentences):
    phraseSentences = Phrases(Sentences, min_count = 30, progress_per = 10000)[Sentences]
    Model = FastText(max_n = 4, min_count = 5, min_n = 1, vector_size = 300, window = 5, workers = 4)
    
    #Building Vocabulary
    Model.build_vocab(phraseSentences)

    # train on our data
    Model.train(phraseSentences, epochs = 100, total_examples = len(Sentences))

    return Model.wv

def Vectorizer(Sentence, WVs, LSA):
    if LSA:
        wordsVecs = LSI.fit_transform(csr_matrix([WVs[Word] for Word in Sentence if Word in WVs]))
    else:
        wordsVecs = [WVs[Word] for Word in Sentence if Word in WVs]
        
    if len(wordsVecs) == 0:
        return zeros(300)
    
    wordsVecs = array(wordsVecs)
    return wordsVecs.mean(axis = 0)

def suppVec(WVs, LSA):
    SVMX_train, SVMX_test = array([Vectorizer(Sentence, WVs, LSA) for Sentence in Sentences]), array([Vectorizer(Sentence, WVs, LSA) for Sentence in Sentences])
    
    # Train a classification model
    SVM.fit(SVMX_train, y_train)

    y_pred = SVM.predict(SVMX_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}\nPrecision: {precision_score(y_test, y_pred, pos_label = 1)}\nRecall: {recall_score(y_test, y_pred, pos_label = 1)}\nF1 score: {f1_score(y_test, y_pred, pos_label = 1)}\n{classification_report(y_test, y_pred)}")

    # Generate scatter plot for training data
    scatter(SVMX_train[:,0], SVMX_train[:,1])
    title("Linearly Separable Data")
    xlabel("X1")
    ylabel("X2")
    show()
    
    # Get support vectors
    supportVectors = SVM.support_vectors_

    # Visualize support vectors
    scatter(SVMX_train[:,0], SVMX_train[:,1])
    scatter(supportVectors[:,0], supportVectors[:,1], color = "red")
    title("Linearly Separable Data with Support Vectors")
    xlabel("X1")
    ylabel("X2")
    show()

    Title = "Support Vector Machine Model Confusion Matrix"
    ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = ["Positive", "Negative"]).plot()
    title(Title)
    savefig(Title, dpi = 1200)
    show()

def NB(tfidf):
    # Training by Naive Bayes classifier method.
    if tfidf:
        # tf-idf vectorizing approach.
        NBX_train, NBX_test = tfVectorizer.fit_transform(X_train), tfVectorizer.transform(X_test)
    else:
        vectorizerFit = tfVectorizer.fit(X_train)
        NBX_train, NBX_test = vectorizerFit.transform(X_train), vectorizerFit.transform(X_test)

    NBClassifier.fit(NBX_train, y_train)

    # Reporting accuracy.
    y_pred, Title = NBClassifier.predict(NBX_test), "Naive Bayes Model Precision-Recall Curve"
    print(f"Harmonic mean (F1 score) for Naive Bayes is {f1_score(y_test, y_pred)}\nAccuracy score for Naive Bayes is {accuracy_score(y_test, y_pred)}\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Plotting Precision-Recall Curve
    Precision, Recall, Threshold = precision_recall_curve(y_test, NBClassifier.predict_proba(NBX_test)[:,1])
    fig, ax = subplots(figsize = (6, 6))
    ax.plot(Recall, Precision, label = "Naive Bayes Classification", color = "firebrick")
    ax.set_title(Title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    box(False)
    ax.legend()
    title(Title)
    savefig(Title, dpi = 1200)
    show()

    Title = "Naive Bayes Model Confusion Matrix"
    ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = ["Positive", "Negative"]).plot()
    title(Title)
    savefig(Title, dpi = 1200)
    show()
    
"""
# Just a try to plotting what happens with Word2Vec and GloVe word embedding methods!
def Plot2Vec(model, num):
    # Defining a tsne function to visualize
    labels, tokens, x, y = [], [], [], []
    for word in model: 
        try:
            tokens.append(model[word])
        except KeyError:
            word = Approx(word, Vocabulary)
            tokens.append(model[word])
        labels.append(word)
    tsne = TSNE(perplexity = 40, n_components = 2, init = 'pca', n_iter = 2500, random_state = 23)
    data = tsne.fit_transform(tokens[:num])
    for each in data:
        x.append(each[0])
        y.append(each[1])
    plt.figure(figsize = (10, 10))
    
    for i in range(num):
        scatter(x[i], y[i])
        annotate(labels[i], xy = (x[i], y[i]), xytext = (5,2), textcoords = 'offset points', ha = 'right', va = 'bottom')
    show()

def PloVe(model, num):
    labels, tokens, x, y = [], [], [], []
    for word in model.wv.vocab:
        tokens.append(glove.word_vectors[glove.dictionary[word]])
        labels.append(word)
    tsne = TSNE(perplexity = 40, n_components = 2, init = 'pca', n_iter = 2500, random_state = 23)
    data = tsne.fit_transform(tokens[:num])
   
    for each in data:
        x.append(each[0])
        y.append(each[1])
    plt.figure(figsize = (10, 10))
    for i in range(300):
        scatter(x[i], y[i])
        annotate(labels[i], xy = (x[i], y[i]), xytext = (5,2), textcoords = 'offset points', ha = 'right', va = 'bottom')
    title('Word vectorization using Glove')
    show()
"""

# Globals!
tfVectorizer, tfidf, NBClassifier, wordEmbeddings, docEmbeddings, SVM, LSI, posTrain, negTrain, posTest, negTest, X_train, X_test, y_train, y_test, Sentences = CountVectorizer(), TfidfVectorizer(), MultinomialNB(), {WordVec: "Word2Vec", GloVe: "GloVe", fastText: "FastText"}, ["DV", "WV"], SVC(C = 1.0, kernel = 'linear', degree = 3, gamma = 'auto'), TruncatedSVD(n_components = 2, algorithm="arpack"), listdir("train/pos"), listdir("train/neg"), listdir("test/pos"), listdir("test/neg"), [], [], [], [], []
numberDocuments = min(len(posTrain), len(negTrain), len(posTest), len(negTest))
# Label '1' for positive and label '0' for negative.
for Index in range(numberDocuments):
    # Train data
    Pos, Neg = preProcessor(open(f"train/pos/{posTrain[Index]}", "r").read()), preProcessor(open(f"train/neg/{negTrain[Index]}", "r").read())
    Sentences += [Pos, Neg]

    # Updating dataframes as lists
    X_train, X_test, y_train, y_test = X_train + [" ".join(Pos), " ".join(Neg)], X_test + [" ".join(preProcessor(open(f"test/pos/{posTest[Index]}", "r").read())), " ".join(preProcessor(open(f"test/neg/{negTest[Index]}", "r").read()))], y_train + [1, 0], y_test + [1, 0]

# Let's start!
print(f"A good approximation for term-document with tf-idf metric is:\n{term_doc_mat_LSA()}\nHere's Naive Bayes classifiying results trained by train documents and labels and evaluated by being test on test documents and labels.\n{NB(False)}\nAnd with tf-idf approach:\n{NB(True)}")

for wordEmbedding in wordEmbeddings: # Check evaluations for each wordembedding method
    vectorWords, embeddingTechnique = wordEmbedding(Sentences), wordEmbeddings[wordEmbedding]
    print(f"\nWith {embeddingTechnique} word embedding technique.")
    while True:
        docEmbedding = input("\nWanna transform documnets to vectors strictly or with word embedding vectors? <DV, WV> ").upper()
        if docEmbedding in docEmbeddings:
            print("\nAlright")
            break
        print("\nChoose one, boy!")

    if docEmbedding == "DV":
        vectorsDocs = DocVec()
    else:
        Vocabulary, Starter, vectorsDocs = list(vectorWords.key_to_index.keys()), array([0] * 300), []
        for Document in Sentences:
            documentVector = Starter
            for Word in Document:
                try:
                    documentVector = add(documentVector, vectorWords[Word])
                except KeyError:
                    documentVector = add(documentVector, vectorWords[Approx(Word, Vocabulary)])

            vectorsDocs.append(documentVector)

    print(f"\nHere's Raw Support Vector Machine classifying with {embeddingTechnique} form embedded word vectors.\n{suppVec(vectorWords, False)}\nAnd now, SVM according to Latent Semantic Analysis!\n{suppVec(vectorWords, True)}\n")

print("Phew...")
