import warnings

warnings.filterwarnings('ignore')

import spacy
import pandas as pd
import numpy as np
import re
from spacy.util import minibatch, compounding
from spacy.tokens import DocBin
from spacy.training import Example
import random
import matplotlib.pyplot as plt

# Second Task: Take a basic, pretrained NER model, and train further on a task-specific dataset



class NerModel:
    
    # This function is to show the performance of a basic NER model using our dataset
    def model_example(data_train,nlp):

        count = 0

        for count,sentence in enumerate(data_train['review']):
            if count < 10:
                break
                
            doc = nlp(sentence)
            entities=[]
                
            for word in doc.ents:
                if word.label_ not in ['DATE','ORDINAL','CARDINAL', 'TIME']:
                    entities.append((word.text, word.label_))   
                
            print(entities)
            
    # I am creating a function that creates a new sentences just using alphanumeric characters in lowercase son it doesn't mess with the model
    def create_new_review(review):
        
        new_review = []

        for token in review.split():
            token = ''.join(char.lower() for char in token if char.isalnum())
            new_review.append(token)

        return ' '.join(new_review)
    
    
    # Now it's time for creating the training data
    def create_data_train(data_train,drugs):
        
        # To create a model spacy needs an array that contains the text and the entities mentioning where they begin and where they end + the tag
        # REMINDER: Dataset has 161,297 entries  

        TRAIN_DATA = []
        PERCENT_OF_DATASET_TO_TRAIN = 1 # Change so it doesn't take that long

        sample = len(data_train)*PERCENT_OF_DATASET_TO_TRAIN
        
        num_drugs=len(drugs)

        count = 0 
        for _, item in data_train.iterrows():
            entities_dict = {} #entities dictionary

            if count <= sample:
                review_up = NerModel.create_new_review(item['review']) # Using the function create_new_review

                learn_items = [] # basically the things it already knows
                entities = [] # the entities position and tag

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
                    
        return TRAIN_DATA
    
    # Next I create function that trains our model
    def train_model(TRAIN_DATA,n_iterations):
            
        loss_new=[]
            
        nlp = spacy.load("en_core_web_sm") # It starts from an already created model

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
            batch_size = 100  


            for itn in range(n_iterations):
                    
                random.shuffle(TRAIN_DATA)
                losses={}
                    
                for batch in spacy.util.minibatch(TRAIN_DATA, size=batch_size):
                        
                    examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
                        
                    losses = nlp.update(examples, drop=0.2, sgd=optimizer)
                        
                        
                        
                
                    '''loss_new.append(losses.get('ner'))'''
                    
                # Creating a graph that shows the losses
                
                '''plt.plot(loss_new, 'palevioletred', label='Training loss')
                plt.title('Training loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()'''
                
        return(nlp)
    
    def test_model(data_test,nlp_model,drugs):
        
        test_reviews = data_test['review']
        # test_reviews = data_test.iloc[-10:, :]['review']
        
        count=0

        # A funcion that  will print the parragraph and the entity that it found
        for review in test_reviews:
            
            new_review = NerModel.create_new_review(review)
            doc = nlp_model(new_review)
            entity =[]
            
            
            for ent in doc.ents:
                if [(ent.text, ent.label_)] != []:
                    #print(new_review)
                    #print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
                    #print('________________________________________________________')
                    entity.append([[ent.text, ent.label_] for ent in doc.ents])
                    count +=1
        
                #print('The model tagged ',count,' reviews that contained drugs')
                #print('________________________________________________________')
            
        NerModel.model_efficiency(count,drugs,entity)
            
    def model_efficiency(count,drugs,entity):
            
        texts=[]
        error=0
        
        accuracy =[]
        for i in entity:
            texts.append(entity[0]) 
            
        for word in texts:
            if word not in drugs:
                error += 1
            
            error = ((error*100)/count)*100
            a_rate = (1-((error*100)/count))*100
            accuracy.append(a_rate)
            
        #print ('The model has a ',f"{ error_rate}%",' error')
        #print('________________________________________________________')
        #print ('The model has a ',f"{ a_rate}%",' acurracy')
        #print('________________________________________________________')
        
     
        
            print('Accuracy: ',accuracy.pop())
        
        
            
            
    def test_ner():

        data_train = pd.read_csv('drugsTrain.csv').fillna('')
        data_test = pd.read_csv('drugsTest.csv').fillna('') #The test dataset
        
        data_test=data_test.replace('&#039;', "'")
        data_test=data_test.replace('039', "'")

        # This dataset contains 161,297 entries 

        drugs = data_train['drugName'].value_counts().index.tolist()

        # There are 3436 drugs in this dataset

        # Testing a trained model of spacy 
        
        nlp = spacy.load('en_core_web_sm')
        
        NerModel.model_example(data_train,nlp)
        
        # It doesn't tag the drugs correctly because it is not trained for that
        
        # In order to use spacy, the data has to be organized in a cetain way so we are using the following function
        
        TRAIN_DATA = NerModel.create_data_train(data_train,drugs)
        
        # Next we use the function that trains our model
        
        nlp_trained = NerModel.train_model(TRAIN_DATA, 50)
        nlp_trained.to_disk('drugs_model')
        
        # Finally we test the model
        
        NerModel.test_model(data_test,nlp_trained,drugs)
        

NerModel.test_ner()