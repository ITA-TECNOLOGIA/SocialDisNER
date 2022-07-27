import unittest
import os
import config 

import pandas as pd

from disease_classifier.filter.zero_shot_classifier import ZeroShotClassifier

ABBREVIATIONS_FILE_PATH = os.path.join(config.PROJECT_PATH, 'dictionaries/siglas_medicas_es.csv') 
DICTIONARY_FILE_PATH = os.path.join(config.PROJECT_PATH, '<Dictionary file path>')

class TestFilterZeroShot(unittest.TestCase):

    def test_filter_abbreviations(self):
        abbreviations_df = pd.read_csv(ABBREVIATIONS_FILE_PATH, header=None, sep=';', names=['abv', 'def'])
        abbreviations_dict = {row[1]:row[2] for row in abbreviations_df[['abv', 'def']].itertuples()}
        abbreviation_diseases = dict()

        filter_classifier = ZeroShotClassifier(
                ["enfermedad", "sociedad", "cultura", "cocina", "deportes"]
            )

        for abbreviation, definition in abbreviations_dict.items():
            classify_disease = filter_classifier.filter_terms(definition, threshold=0.4)
            if classify_disease:
                abbreviation_diseases[abbreviation] = definition

        output_df = pd.DataFrame.from_dict(abbreviation_diseases.items())
        output_df.to_csv(os.path.join(config.PROJECT_PATH, '<Output file path>'), sep=";", header=None, index=False)

    def test_filter_dictionary(self):
        dictionary_df = pd.read_csv(DICTIONARY_FILE_PATH, header=None, sep=';')
        dictionary_dict = dictionary_df.drop_duplicates().to_dict('records')
        print(f"Original dictionary length: {len(dictionary_dict)}")
        filtered_dictionary = dict()
        filter_classifier = ZeroShotClassifier(
                ["enfermedad", "sociedad", "cultura", "cocina", "deportes"]
            )

        for elem in dictionary_dict[1:]:
            classify_disease = filter_classifier.filter_terms(elem[0], threshold=0.4) #0.35
            if classify_disease:
                filtered_dictionary[elem[0]] = elem[1]

        print(f"Filtered dictionary length: {len(dictionary_dict)}")
        output_df = pd.DataFrame.from_dict(filtered_dictionary.items())
        output_df.to_csv(os.path.join(config.PROJECT_PATH, '<Output file path>'), sep=";", header=None, index=False)