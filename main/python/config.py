from logging_ita import configure
import os
import re
import string

logger = configure()

PROJECT_PATH = "<Set here base project path>"
CORPUS_FOLDER = "corpus"
MODELS_FOLDER = "models"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  

WANDB_PROJECT_NAME = "disease_token_classifier_project"

LIST_ORIGINAL_PRETRAINED_MODELS = [
    "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
    "PlanTL-GOB-ES/roberta-large-bne-capitel-ner",
    "Babelscape/wikineural-multilingual-ner",
    "bertin-project/bertin-base-ner-conll2002-es",
    "PlanTL-GOB-ES/roberta-large-bne",
    "PlanTL-GOB-ES/bsc-bio-es",
    "PlanTL-GOB-ES/bsc-bio-ehr-es",
    "PlanTL-GOB-ES/bsc-bio-ehr-es-cantemist",
    "cardiffnlp/twitter-xlm-roberta-base",
]

LIST_FINETUNED_MODELS = [
    "<Paths of finetuned models for prediction>",
    "..."
    ]

TRAIN_DATASET_V4_PATH = "<Train csv file path>" 
VALIDATION_DATASET_V4_PATH = "<Validation csv file path>"
TRAIN_VALID_PATHS = {
    "train": TRAIN_DATASET_V4_PATH,
    "validation": VALIDATION_DATASET_V4_PATH,
}

TEST_DATASET_PATH = "<Train csv file path>"


ZERO_SHOT_MODEL = "Recognai/bert-base-spanish-wwm-cased-xnli"

GAZETTEER_DISTEMIST = ".../resources/distemist/dictionary_distemist.tsv"

VAL_GAZETTEER_ES_PATH = "<Validation gazetteer (reduced version)>"
VAL_GAZETTEER_ES_FULL_PATH = "<Validation gazetteer (final version)>"
TEST_GAZETTEER_ES_PATH= "<Test gazetteer (reduced version)>"
TEST_GAZETTEER_ES_FULL_PATH = "<Test gazetteer (final version)>"

OUTPUT_PATH = "<Set here default output file path>"

string_puntuaction = re.escape(f"{string.punctuation}“””¿´,")
pattern_any_puntuaction_mark = re.compile(rf"[{string_puntuaction}]+")
pattern_punctuation_start = re.compile(f"^[{string_puntuaction}]")
pattern_punctuation_end = re.compile(f"[{string_puntuaction}]+$")
substitude_expresion = re.compile(f"[{string_puntuaction}]")
extract_initials_expresion = re.compile("([A-Z]+[0-9]*)")
extract_camel_case = re.compile("([A-Z][^A-Z][^A-Z]+)")
hashtag_mention_regex = re.compile("(#[A-Za-z0-9_]+)|(@[A-Za-z0-9_]+)")
integers_regex = re.compile("^[-+]?[0-9]+$")

# Model Parameters
VERSION = 4
BATCH_SIZE = 4 #8
EPOCH_SIZE = 5
LEARNING_RATE = 3e-5 #2e-5
PATIENCE = 6 # EarlyStopping parameter
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
NUMBER_OF_STEPS=5000 
CUDA_ID=1  

def remove_tildes(text):
    """
    Removes tildes
    Input: text. Output: preprocessed text
    """
    a, b = "áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU"
    trans = str.maketrans(a, b)
    return text.translate(trans)

