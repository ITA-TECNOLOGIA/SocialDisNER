from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

class DiseaseNerPredictor():

    def __init__(self, model_path, cuda_id=-1):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.token_classifier = pipeline(
            "token-classification", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="first", device=cuda_id
        )

    def ner_predict(self, sentence, tweet_id):
        """
        Extracts entities from text using the finetuned model desired
        """
        sentence = sentence.strip()
        entities_list = list()
        prediction_list = self.token_classifier(sentence)
        for token_classification in prediction_list:
            entities_list.append(
                {
                    'tweets_id': tweet_id,
                    'begin': token_classification.get('start'),
                    'end': token_classification.get('end'),
                    'type': token_classification.get('entity_group'),
                    'extraction': token_classification.get('word').replace("\n", " ").strip() 
                })

        return entities_list


