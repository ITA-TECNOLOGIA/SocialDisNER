import unittest
import os
import config 
import pandas as pd
from disease_classifier.transformers.disease_ner_predictor import DiseaseNerPredictor


VALIDATION_FILEPATH = config.VALIDATION_DATASET_V4_PATH
VALIDATION_DF = pd.read_csv(VALIDATION_FILEPATH)

LIST_MODELS = config.LIST_FINETUNED_MODELS

class TestPredictTransformers(unittest.TestCase):

    def test_predict_transformers(self):
        tweet_id = 1263897631295787014
        text = VALIDATION_DF[VALIDATION_DF['tweets_id'] == tweet_id]['text'].values[0]
        print(text)
        
        diseases_trf = DiseaseNerPredictor(model_path=LIST_MODELS[0])
        output = diseases_trf.ner_predict(text, tweet_id)
        print(output)
        self.assertIsNotNone(output)

    def test_predict_transformers_df(self):
        tweet_id = 1263897631295787014
        text = VALIDATION_DF[VALIDATION_DF['tweets_id'] == tweet_id]['text'].values[0]
        print(text)

        diseases_trf = DiseaseNerPredictor(model_path=LIST_MODELS[0])
        output = diseases_trf.ner_predict(text)
        output_df = pd.DataFrame.from_dict(output)
        #output_df = output_df.rename(columns={'entity':'type', 'word':'extraction', 'start':'begin'})
        #output_df = output_df[['begin', 'end', 'type', 'extraction']] #['tweets_id', 'begin', 'end', 'type', 'extraction']
        output_file_path = os.path.join(config.PROJECT_PATH, 'temp', 'test_predict_transformers_df.tsv')
        output_df.to_csv(output_file_path, sep="\t", index=False)

        self.assertIsNotNone(output_df)

if __name__ == '__main__':
    unittest.main()

