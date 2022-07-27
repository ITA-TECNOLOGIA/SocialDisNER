import unittest
import json

from disease_classifier.gazetteer.dictionary_tweet_parser import DictionaryParser
gazetteer_ner = DictionaryParser()

class TestDictionaryParser(unittest.TestCase):

    def test_dictionary_parser_text(self):
        text = "AUTOESTIMA Y ANSIEDAD!\n \nLa ansiedad, la depresión, son dos trastornos emocionales graves, muy graves, a todos nos pueden llegar en cualquier momento de nuestras vidas y por muchas… https://t.co/xBQpL5M4Ke" # TRAIN -> 1092141153149800449	
        text_id = "0001"
        result = gazetteer_ner.dictionary_parser_es(text, text_id)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        self.assertIsNotNone(result)
