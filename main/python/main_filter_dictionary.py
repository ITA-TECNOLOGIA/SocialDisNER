import os
import config 

import pandas as pd

from disease_classifier.filter.zero_shot_classifier import ZeroShotClassifier

DICTIONARY_FILE_PATH = os.path.join(config.PROJECT_PATH, '<CSV file path>')

def main():
    """
    Test Zero-shot classification
    """
    dictionary_df = pd.read_csv(DICTIONARY_FILE_PATH, header=None, sep=';')
    dictionary_dict = dictionary_df.drop_duplicates().to_dict('records')
    print(f"Original dictionary length: {len(dictionary_dict)}")
    filtered_dictionary = dict()
    filter_classifier = ZeroShotClassifier(
            ["enfermedad", "sociedad", "cultura", "cocina", "deportes"]
        )

    for elem in dictionary_dict[1:]:
        classify_disease = filter_classifier.filter_terms(elem[0], threshold=0.3) #0.35
        if classify_disease:
            filtered_dictionary[elem[0]] = elem[1]

    print(f"Filtered dictionary length: {len(filtered_dictionary)}")
    output_df = pd.DataFrame.from_dict(filtered_dictionary.items())
    output_df.to_csv(os.path.join(config.PROJECT_PATH, '<Output file path>'), sep=";", header=['term', 'subtokens'], index=False)

if __name__ == "__main__":
    main()