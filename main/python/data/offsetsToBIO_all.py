import os
import config
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from spacy.training import offsets_to_biluo_tags # (This works for Spacy 3.3.0)

# Tokenizer
from spacy.lang.es import Spanish  
nlp = Spanish()

# # SocialDisNer
# Merging the data from the .tsv file and the txt's
v3 = 'SocialDisNer_v3'
v4 = 'SocialDisNer_v4'

def join_tsv_with_txt(df, tweets_folder):
    # Get text data
    for index, row in df.iterrows():
        tweets_id = row["tweets_id"]
        text_file = os.path.join(tweets_folder, str(tweets_id) + ".txt")

        with open(text_file, "r") as tf:
            text = tf.read()
            df.at[index, 'text'] = text.strip()

    return df


def offsets_to_BILUO(text, begin, end, type):
    YOUR_DATA = [(text, {"entities": [(int(begin), int(end), type)]})]

    for text, annotations in YOUR_DATA:
        offsets = annotations["entities"]
        doc = nlp(text)
        tags = offsets_to_biluo_tags(doc, offsets)
    
        # Perform your operations over the tokens of the document.
        list_tokens = []
        for token in doc:
            list_tokens.append(token.text)

        return pd.Series([tags, list_tokens])


def merge_BILUO_tags(list_tags):
    merged_tags = []

    for item in list_tags:
        a = list(item)
        result = all(element == "O" for element in a)
        if result:
            merged_tags.append("O")
        else:
            l_in = [s for s in a if s != "O"]
            merged_tags.append(l_in[0])
    return merged_tags


def BILOU_to_BIO(bilou_tags):
    # Transform string to list
    if isinstance(bilou_tags, pd.Series):
        bilou_tags = bilou_tags.tolist()[0]
    elif isinstance(bilou_tags, str):
        bilou_tags = eval(bilou_tags)
        
    BIO_tags = []

    for tag in bilou_tags:
        if tag == 'O':
            BIO_tags.append('O')
        elif tag == 'B-ENFERMEDAD':
            BIO_tags.append('B-ENFERMEDAD')    
        elif tag == 'I-ENFERMEDAD':
            BIO_tags.append('I-ENFERMEDAD')    
        elif tag == 'L-ENFERMEDAD':
            BIO_tags.append('I-ENFERMEDAD')    
        elif tag == 'U-ENFERMEDAD':
            BIO_tags.append('B-ENFERMEDAD')         
        else:
            BIO_tags.append('-') # before: BIO_tags.append('O') --> Seems that there have been some kind of error/misalignment
            
    return BIO_tags

def tweets_to_formatted_df(filepath_ner_mentions, texts_folder, save_output_path=None):
    """
    
    """
    df = pd.read_csv(filepath_ner_mentions, delimiter="\t")
    # Merge MENTIONS with its corresponding texts
    df = join_tsv_with_txt(df, tweets_folder=texts_folder)

    # Funtion to transform offsets to BILUO tags by using the method 'offsets_to_biluo_tags' from Spacy
    df[["BILUO_tag_per_occurrence", "tokens"]] = df.progress_apply(lambda row: offsets_to_BILUO(row["text"], row["begin"], row["end"], row["type"]), axis=1)

    # Some tweets contain multiple occurrences of illnesses, we need to merge thoses tags into a single one
    df = df.groupby('tweets_id').agg(
                                    {'begin': list, 
                                    'end': list, 
                                    'extraction': list,
                                    'text': ['first'],
                                    'tokens': ['first'],
                                    'BILUO_tag_per_occurrence': list  
                                    }).reset_index()
    df['BILUO_tags'] = df['BILUO_tag_per_occurrence'].progress_apply(lambda x: merge_BILUO_tags(list(zip(*x.item()))), axis=1)
    
    # Change tags format from BILOU to BIO
    df['BIO_tags'] = df.progress_apply(lambda row: BILOU_to_BIO(row['BILUO_tags']), axis=1)

    # Print the final DataFrame on terminal
    with pd.option_context('display.max_rows', None,
                        'display.max_columns', None,
                        'display.precision', 3,
                        ):
        print(df.head())
    df.columns = df.columns.droplevel(1)
    if save_output_path:
        df.to_csv(save_output_path, index=False)
    
    return df

def main():
    # Set filepaths
    """
    ### TRAIN
    ner_mentions_train = "training-validation-data/mentions_train.tsv"
    filepath_ner_mentions_train = os.path.join(config.PROJECT_PATH, config.CORPUS_FOLDER, v4, ner_mentions_train) 
    tweets_folder_train = "training-validation-data/train-valid-txt-files/training"
    filepath_tweets_train = os.path.join(config.PROJECT_PATH, config.CORPUS_FOLDER, v4, tweets_folder_train)
    output_train = "BIO/training/SocialDisNer_training.csv"
    filepath_output_train = os.path.join(config.PROJECT_PATH, config.CORPUS_FOLDER, v4, output_train)
    formmated_train_df = tweets_to_formatted_df(filepath_ner_mentions=filepath_ner_mentions_train, texts_folder=filepath_tweets_train, save_output_path=filepath_output_train)
    print(formmated_train_df.info())

    ### VALIDATION
    ner_mentions_val = "training-validation-data/mentions_validation.tsv"
    filepath_ner_mentions_val = os.path.join(config.PROJECT_PATH, config.CORPUS_FOLDER, v4, ner_mentions_val) 
    tweets_folder_val = "training-validation-data/train-valid-txt-files/validation"
    filepath_tweets_val = os.path.join(config.PROJECT_PATH, config.CORPUS_FOLDER, v4, tweets_folder_val)
    output_val = "BIO/validation/SocialDisNer_validating.csv"  
    filepath_output_val = os.path.join(config.PROJECT_PATH, config.CORPUS_FOLDER, v4, output_val) 
    formmated_val_df = tweets_to_formatted_df(filepath_ner_mentions=filepath_ner_mentions_val, texts_folder=filepath_tweets_val, save_output_path=filepath_output_val)
    print(formmated_val_df.info())
    """
    ### ADDITIONAL LARGE
    additional_folder = "additional-large_scale_data"
    additional_mentions = "socialdisner_disease_mentions.tsv"
    filepath_ner_mentions_add = os.path.join(config.PROJECT_PATH, config.CORPUS_FOLDER, v4, additional_folder, additional_mentions) 
    additional_tweets_folder = "tweets_txt"
    filepath_tweets_add = os.path.join(config.PROJECT_PATH, config.CORPUS_FOLDER, v4, additional_folder, additional_tweets_folder)
    output_add =  "BIO/training/additional_data.csv" 
    filepath_output_add = os.path.join(config.PROJECT_PATH, config.CORPUS_FOLDER, v4, output_add) 
    formmated_add_df = tweets_to_formatted_df(filepath_ner_mentions=filepath_ner_mentions_add, texts_folder=filepath_tweets_add, save_output_path=filepath_output_add)
    print(formmated_add_df.info())

    
if __name__ == "__main__":
    main()
