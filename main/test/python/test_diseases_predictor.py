import unittest
import os
import json

import config
import pandas as pd

from config import LIST_FINETUNED_MODELS

from disease_classifier.diseases_predictor import DiseasesPredictor
disease_predictor = DiseasesPredictor(
        LIST_FINETUNED_MODELS[:1],
        include_token_dict=True,
        include_zero_shot_filter=False,
        include_model_predict=True,
        remove_emojis=True,
        is_validation=True,
        gazetter_file=config.VAL_GAZETTEER_ES_PATH,
        cuda_id=0
    )

class TestDiseasesPredictor(unittest.TestCase):

    def test_diseases_predictor_tweet(self):
        tweet_id = 1452329159062065154 
        validation_filepath = os.path.join(config.PROJECT_PATH, config.CORPUS_FOLDER, '<Validation CSV file path>')
        validation_df = pd.read_csv(validation_filepath)
        text = validation_df[validation_df['tweets_id'] == tweet_id]['text'].values[0]
        print(text)
        result = disease_predictor.predict_tweet(text, tweet_id, parallel=False)
        print(result)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()