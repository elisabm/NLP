# **NLP Project**

Natural language processing (NLP) is a branch of artificial intelligence concerned with giving computers the ability to understand text and spoken words in the same way human beings can. NLP mixes rule-based modeling with statistical, machine learning, and deep learning models. As a result a computer gains the ability to process human language. 

This project is separated into three realistic tasks within the NLP universe. Later in this document we'll be talking about each one of them. 

## **Task 1: Out of the Box Sentiment Analysis**

Sentiment analysis is a NLP technique that is used to determine whether data has a positive or negative sentiment. Sentiment analysis focuses on the polarity of a text but it also can detect specific feelings and emotions, urgency and intentions.

In this task we are trying to determine whether a certain movie review has a positive or a negative sentiment. To do this 'siebert/sentiment-roberta-large-english' model from HuggingFace. We are only printing the result of the analysis (POSITIVE or NEGATIVE).

https://monkeylearn.com/sentiment-analysis/

## **Task 2: Use pretrained NER model, and train further on a task-specific dataset**

Named entity recognition (NER) is a sub-task of information extraction (IE) that searches and categorises specified entities in a text. NER is usually used in Natural Language Processing (NLP) and Machine Learning.

For this task, I decided to use an open-source library called SpaCy. SpaCy is used for advanced NLP in python, it helps build applications that process and “understand” large volumes of text.

SpaCy provides efficient statistical system for NER in python, which can assign labels to groups of tokens which are contiguous. It provides a default model which can recognize a wide range of named or numerical entities, which include person, organization, language, event etc. It is important to mention that apart from these default entities, spaCy also gives the liberty to add arbitrary classes to the NER model, by training the model to update it with newer trained examples. This trait of SpaCy is what we are going to use in the project.

The main goal in this task is to use an already trained model and train it further, so in order to achive that we are going to choose a model and a dataset that has information new to our model. 


In this case I used the 'en_core_web_sm' model that SpaCy already has. In my run.py I had added a little section that tries to use our model in the data. And we can observe that it doen't tag the things correctly. To solve this situation we show the model new data so it can learn new things to tag. This chosen dataset contains 161,297 entries of different reviews about medicines. What we want is to teach the model the medicines mentioned in the dataset, in order for it to tag them properly. 





