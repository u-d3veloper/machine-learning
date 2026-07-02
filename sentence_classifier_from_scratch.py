
# Write from scratch (you can only use Numpy arrays) 
# very basic and simple algorithm to classify sentences:

test_dataset = [
    "cats like meat and fish is best for cats",
    "train your mind reading good fiction, thrillers and other books",
    "I love books and reading while seating.",
    "Dogs love cats' food also"
]


# Use these sentences to train your classifier:
dataset = [
    {"class" : 1, "sentence":"meat is a good food for all dogs and cats , dogs also like apples"},
    {"class" : 2, "sentence":"reading fiction books is a good food for mind and some thrillers are not"},
]


######################## CODE HERE #############################
import numpy as np
from typing import List
import re

class CountVectorizer():
    
    def __init__(self):
        self.vocab = {}
        self.oov = []

    def fit(self, sentences : List[str]):
        count = 0
        for sentence in sentences:
            for word in sentence.split(" "):
                if not word in self.vocab.keys():
                    self.vocab[word] = count
                    count+=1
        print(self.vocab)

    def transform(self, sentence : str):
        count_vector = np.zeros(len(self.vocab))
        for word in sentence.split(" "):
            if word in self.vocab.keys():
                count_vector[self.vocab[word]] += 1
            else: self.oov.append(word)
        return count_vector

    def _preprocess(self, sentence:str):
        re.sub(r'[^\w\s]', ' ', sentence)

        words = [word.lower().strip() for word in sentence.split()]

        return " ".join(words)

class Classifier():

    def __init__(self, vectorizer: CountVectorizer, training_data: List[str]):
        self.training_data = training_data
        self.vectorizer = vectorizer
    
    def compute_similarity(self, sentence1:str, sentence2:str):
        comp1 = np.array(self.vectorizer.transform(sentence1))
        comp2 = np.array(self.vectorizer.transform(sentence2))
                        
        dot = np.dot(comp1,comp2)
        mag1 = np.linalg.norm(comp1)
        mag2 = np.linalg.norm(comp2)

        return dot/(mag1*mag2)

    def predict(self, test_sent:str):
        sims = [self.compute_similarity(test_sent, sent) for sent in self.training_data]
        best_index = np.argmax(sims)
        best_similarity = sims[best_index]
        return [best_index+1, best_similarity]

X_train = [sentence["sentence"] for sentence in dataset]
vectorizer = CountVectorizer()
vectorizer.fit(X_train)

classifier = Classifier(vectorizer, X_train)


for sentence in test_dataset:
    print(f'La phrase "{sentence}" appartient à la classe :{classifier.predict(sentence)[0]}')

