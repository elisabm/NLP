
# Third Task
import translators as ts
from itertools import islice
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

#Third Task: Set up and compare model performance of two different translation models

class TranslationModel:
  
  def translate(file_to_translate,translated_file,n_lines):

      # Opening both files and using only the first 100 lines
      with open(file_to_translate) as myfile:
          esp = list(islice(myfile, n_lines))

      esp = [x.replace("\n", "") for x in esp]

      with open(translated_file) as myfile:
          eng = list(islice(myfile, n_lines))
          
      eng = [x.replace("\n", "") for x in eng]

      # With the library translators we can call different translators

      es_to_en_google=[]

      # Note: we need to do sentence by sentence because the library doesn't accept the whole paragraph, this is due to it's lenght.
      # Here we are using Google
      for i in range(len(esp)):
        res = ts.google(esp[i], to_language='en')
        es_to_en_google.append(res)



      es_to_en_baidu=[]

      # Here we are using Baidu
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

TranslationModel.translate('es_corpus.txt','en_corpus.txt',100)

