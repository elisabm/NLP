
#First Task: Out of the Box Sentiment Analysis 

from transformers import pipeline

#I used a model that I found on HugginFace, according to it's documentation the model is really effective

model = pipeline('sentiment-analysis',model="siebert/sentiment-roberta-large-english")

with open('tiny_movie_reviews_dataset.txt') as f:
   lines = f.readlines()

#Printing my results 

for line in lines:
    var = model(line)
    print (var[0]['label']) 
    

#Second Task: Take a basic, pretrained NER model, and train further on a task-specific dataset

import spacy
import pandas as pd
import numpy as np
import re
from spacy.util import minibatch, compounding
from spacy.tokens import DocBin
from spacy.training import Example
import random
import matplotlib.pyplot as plt

data_train = pd.read_csv('drugsTrain.csv').fillna('')

#Dataset contains 161,297 entries 

drugs = data_train['drugName'].value_counts().index.tolist()

#There are 3436 drugs in this dataset

#Testing a trained model of spacy 

nlp=spacy.load('en_core_web_sm')

count = 0

for sentence in data_train['review']:
  if count <= 10:
      
      doc = nlp(sentence)
      entities=[]
      
      for word in doc.ents:
        if word.label_ not in ['DATE','ORDINAL','CARDINAL', 'TIME']:
          entities.append((word.text, word.label_))
          
      count +=1
      
      print(entities)

#It doesn't tag the drugs correctly because it is not trained for that


#I am creating a function that creates a new senteces just using alphanumeric characters in lowercase son it doesn't mess with the model
def newreview(review):
    new_review = []

    for token in review.split():
        token = ''.join(char.lower() for char in token if char.isalnum())
        new_review.append(token)

    return ' '.join(new_review)



#Now it's time for creating the training data 

#To create a model spacy needs an array that contains the text and the entities mentioning where they begin and where they end + the tag
count = 0 
TRAIN_DATA = []

#REMINDER: Dataset has 161,297 entries  


PERCENT_OF_DATASET_TO_TRAIN = 0.2 #Change so it doesn't take that long

sample = len(data_train)*PERCENT_OF_DATASET_TO_TRAIN


for _, item in data_train.iterrows():
    entities_dict = {} #entities dictionary

    if count < sample:
        review_up = newreview(item['review']) #Using the function above

        learn_items = [] # basically the things it already knows
        entities = [] #the entities position and tag

        for word in review_up.split():
            if word in drugs:

                for i in re.finditer(word, review_up):
                    if word not in learn_items:

                        entity = (i.span()[0], i.span()[1], 'DRUG')
                        learn_items.append(word)
                        entities.append(entity)

        if len(entities) > 0:
            entities_dict['entities'] = entities
            
            train_item = (review_up, entities_dict)
            TRAIN_DATA.append(train_item)

        count+=1


#Next I create function that trains our model

def train(n_iterations):
    
  loss_new=[]
 
  nlp = spacy.load("en_core_web_sm") #It starts from an already created model

  if "ner" not in nlp.pipe_names:
      ner = nlp.add_pipe("ner", last=True)
  else:
      ner = nlp.get_pipe("ner")


  for _, annotations in TRAIN_DATA:

    for ent in annotations.get('entities'):
      ner.add_label(ent[2])
  
  losses = None

  other_pipes =[pipe for pipe in nlp.pipe_names if pipe != 'ner']
  
  with nlp.disable_pipes(*other_pipes):
      
    optimizer = nlp.create_optimizer()
    optimizer.learn_rate = 0.001  
    batch_size = 32  


    for itn in range(n_iterations):
        
        random.shuffle(TRAIN_DATA)
        losses={}
        
        for batch in spacy.util.minibatch(TRAIN_DATA, size=batch_size):
            
            examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
            
            losses = nlp.update(examples, drop=0.2, sgd=optimizer)
            
            
            
       
        loss_new.append(losses.get('ner'))
        
    #Creating a graph that shows the losses
    
    plt.plot(loss_new, 'palevioletred', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
  
  return(nlp)


nlp = train(30) # I used 30 iterations
nlp.to_disk('drugs_model')


data_test = pd.read_csv('drugsTest.csv').fillna('') #The test dataset

test_reviews = data_test.iloc[-10:, :]['review']

#A funcion that  will print the parragraph and the entity that it found
for review in test_reviews:
    
    review = newreview(review)
    print(review)
    doc = nlp(review)

    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print('________________________________________________________')


#Third Task: Set up and compare model performance of two different translation models

import translators as ts
from itertools import islice
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

#Opening both files and using only the first 100 lines
with open('es_corpus.txt') as myfile:
    esp = list(islice(myfile, 100))

esp = [x.replace("\n", "") for x in esp]

with open('en_corpus.txt') as myfile:
    eng = list(islice(myfile, 100))
    
eng = [x.replace("\n", "") for x in eng]

#With the library translators we can call different translators

es_to_en_google=[]

#Note: we need to do sentence by sentence because the library doesn't accept the whole paragraph, this is due to it's lenght.
#Here we are using Google
for i in range(len(esp)):
  res = ts.google(esp[i], to_language='en')
  es_to_en_google.append(res)



es_to_en_baidu=[]

#Here we are using Baidu
for i in range(len(esp)):
  res = ts.google(esp[i], to_language='en')
  es_to_en_baidu.append(res)
  

results_google = []
for sentence in es_to_en_google:
    sentence_results = []
    for s in sentence:
        sentence_results.append(nltk.word_tokenize(sentence))
    results_google.append(sentence_results)

results_baidu = []
for sentence in es_to_en_baidu:
    sentence_results = []
    for s in sentence:
        sentence_results.append(nltk.word_tokenize(sentence))
    results_baidu.append(sentence_results)
    

results_eng = []
for sentence in eng:
    sentence_results = []
    for s in sentence:
        sentence_results.append(nltk.word_tokenize(sentence))
    results_eng.append(sentence_results)
    
list_of_google_references = [results_google] 
list_of_baidu_references = [results_baidu]
list_of_hypotheses = [results_eng] # list of hypotheses that corresponds to list of references.

chencherry = SmoothingFunction()


google_score =nltk.translate.bleu_score.corpus_bleu(es_to_en_google, eng, smoothing_function=chencherry.method1,weights=(0.25, 0.25, 0.25, 0.25))
baidu_score =nltk.translate.bleu_score.corpus_bleu(es_to_en_baidu, eng, smoothing_function=chencherry.method1,weights=(0.25, 0.25, 0.25, 0.25))


print('BLEU score Google -> ', google_score)
print('BLEU score Baidu -> ', baidu_score)

