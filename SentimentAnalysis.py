from transformers import pipeline 


# First Task: Out of the Box Sentiment Analysis 

# I used a model that I found on HuggingFace, according to its documentation the model is really effective


class SentimentAnalysis:
           
    def analysis(database,model_internet):
        
        model = pipeline('sentiment-analysis',model= model_internet)
        
        with open(database) as f:
            text = f.readlines()

        # Printing my results 

        for line in text:
            result = model(line)
            print(result[0]['label']) 
            
        return 
        
        
SentimentAnalysis.analysis('tiny_movie_reviews_dataset.txt',"siebert/sentiment-roberta-large-english")


