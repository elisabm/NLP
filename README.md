# **NLP Project**

Natural language processing (NLP) is a branch of artificial intelligence concerned with giving computers the ability to understand text and spoken words in the same way human beings can. NLP mixes rule-based modeling with statistical, machine learning, and deep learning models. As a result a computer gains the ability to process human language. 

This project is separated into three realistic tasks within the NLP universe. Later in this document we'll be talking about each one of them. 

## **Task 1: Out of the Box Sentiment Analysis**

Sentiment analysis is a NLP technique that is used to determine whether data has a positive or negative sentiment. Sentiment analysis focuses on the polarity of a text but it also can detect specific feelings and emotions, urgency and intentions.

In this task we are trying to determine whether a certain movie review has a positive or a negative sentiment. To do this 'siebert/sentiment-roberta-large-english' model from HuggingFace. We are only printing the result of the analysis (POSITIVE or NEGATIVE).

The results of the code print out the following:
<p align="center">
  
  <img  src="https://user-images.githubusercontent.com/112834283/201458928-5a54e045-4137-4ccc-bf8f-a9a61be6ee2e.png">
  
</p>

## **Task 2: Use pretrained NER model, and train further on a task-specific dataset**

Named entity recognition (NER) is a sub-task of information extraction (IE) that searches and categorises specified entities in a text. NER is usually used in Natural Language Processing (NLP) and Machine Learning.

For this task, I decided to use an open-source library called SpaCy. SpaCy is used for advanced NLP in python, it helps build applications that process and “understand” large volumes of text.

SpaCy provides efficient statistical system for NER in python, which can assign labels to groups of tokens which are contiguous. It provides a default model which can recognize a wide range of named or numerical entities, which include person, organization, language, event etc. It is important to mention that apart from these default entities, spaCy also gives the liberty to add arbitrary classes to the NER model, by training the model to update it with newer trained examples. This trait of SpaCy is what we are going to use in the project.

The main goal in this task is to use an already trained model and train it further, so in order to achive that we are going to choose a model and a dataset that has information new to our model. 

In this case I used the 'en_core_web_sm' model that SpaCy already has. In my run.py I had added a little section that tries to use our model in the data. And we can observe that it doen't tag the things correctly, like seen below:

<p align="center">
  
  <img  src="https://user-images.githubusercontent.com/112834283/201459001-b0657add-e717-4108-b6d4-70b0a3be3965.png">
  
</p>


To solve this situation we show the model new data so it can learn new things to tag. This chosen dataset contains 161,297 entries of different reviews about medicines. What we want is to teach the model the medicines mentioned in the dataset, in order for it to tag them properly. 

First step is creating data that SpaCy can recieve, meaning we need to transform the dataset to the following format: 

array that contains the text and the entities mentioning where they begin and where they end plus the tag. 

Next we use our model 'en_core_web_sm' and basically just train it with our dataset. *NOTE: this process is very long because we have more than 161,000 entries*

Here we print our training loss with the number of iterations. With the whole dataset the graph looks something like this:

<p align="center">
  
 
  <img  src="https://user-images.githubusercontent.com/112834283/201459370-fc45bc3d-cf8a-4123-9306-7f22241ab740.png">
  
</p>


Finally we try it in new data (there is a specific dataset for that), the result is going to print something like this:

<p align="center">
  
 <img  src="">
  
</p>

It is a sentence and our model found a medicine and tagged it correctly. 

## **Task 3: Set up and compare model performance of two different translation models**

Translation models can be used to build conversational agents across different languages. It is important in the NLP field because of the human need to understand different languages. 

Here we're not going to use it in a production level, we're just going to translate 100 lines of a given text (es_corpus.txt). In order to solve this task in the easiest way possible I searched for a library that doesn't require API keys and I found it. 

The library mentioned above is *Translators* ,it supports numerous engines, including Google, DeepL, Baidu, and others. You can find the documentation here: https://github.com/UlionTse/translators. I installed it and thendecided which translation engine to use (Google and Baidu). To use an engine, you simply need to call a method with the corresponding name, in the code it is similar to this ts.google(text, to_language='en'). 

My Bleu results are this:

<p align="center">
  
 <img  src="">
  
</p>
