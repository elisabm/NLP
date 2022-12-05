from SentimentAnalysis import SentimentAnalysis

from NerModel import NerModel

from TranslationModel import TranslationModel


#FIRST TASK

SentimentAnalysis.analysis('tiny_movie_reviews_dataset.txt',"siebert/sentiment-roberta-large-english")

#SECOND TASK

NerModel.test_ner()

#THIRD TASK

TranslationModel.translate('es_corpus.txt','en_corpus.txt',100)