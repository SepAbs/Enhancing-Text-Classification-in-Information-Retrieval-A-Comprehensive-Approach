import nltk
from nltk.tokenize import wordpunct_tokenize
#nltk.download('stopwords')
from nltk.corpus import stopwords
from string import punctuation
from os import listdir

"""
This code put all preprocessed sentences (1000 first) of train set in a text file by being run once.
"""

def preProcessor(String):
    null, Consonants, Irregulars, stopWords = "", "qwrtypsdfghjklzxcvbnm", {"begun": "begin", "frozen": "freeze", "children": "child", "feet": "foot", "teeth": "tooth","mice": "mouse", "people": "person"}, stopwords.words('english')
    String = wordpunct_tokenize(String) # Case Folding & Tokenisations
    String = [Token.lower() for Token in String if Token not in punctuation and Token.lower() not in stopWords]
    lengthString = len(String)
    
    for Index in range(lengthString):
        # Asymmetric Expansion
        if String[Index].endswith("ies") and String[Index][-4] in Consonants:
            String[Index] = String[Index][:-3] + "y"

        elif String[Index].endswith("ves"):
            String[Index] = String[Index][:-3] + "f"
    
        elif String[Index].endswith("s"):
            String[Index] = String[Index][:-1]
            
        elif String[Index].endswith("men"):
            String[Index] = String[Index][:-3] + "man"
 
    while null in String:
        String.remove(null)

    return String

posTrain, negTrain, posTest, negTest, trainedDocs, testDocs, trainedLabels, testLabels, Sentences, trainedTexts = listdir("train/pos"), listdir("train/neg"), listdir("test/pos"), listdir("test/neg"), [], [], [], [], [], ""

#Bound the number of trained and test data by 5000.
for Index in range(1000):
    # For GloVe embedding
    trainedTexts += " ".join(preProcessor(open(f"train/pos/{posTrain[Index]}", "r").read())) + " ".join(preProcessor(open(f"train/neg/{negTrain[Index]}", "r").read()))

with open('Continuously Trained Sentenes.txt', 'w+') as fh:
    fh.write(trainedTexts)
