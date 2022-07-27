
import config
import sys
import ast
from token_classifier.token_classifier_optimized import TokenClassifierOptimized
import os

class DiseaseNER(TokenClassifierOptimized):
    """
        Transformer-based token classifier for Diseases recognition
    """
    def __init__(
            self,
            model_pretrained,
            number_of_steps=10000,
            cuda_id=0, 
            wandb_project_name="disease_token_classifier_project"):

        id2label= {
            "0": 'O', 
            "1": 'B-ENFERMEDAD',
            "2": 'I-ENFERMEDAD',
        }
        label2id = {
                'O' : 0,
                'B-ENFERMEDAD': 1,
                'I-ENFERMEDAD': 2,
            }

        label_list =['O', 'B-ENFERMEDAD', 'I-ENFERMEDAD']
        converters_col={'tokens':ast.literal_eval,'BIO_tags':ast.literal_eval}

        super().__init__(
            model_path=os.path.join(config.PROJECT_PATH, config.MODELS_FOLDER),
            dataset_path=config.TRAIN_VALID_PATHS,
            converters_col=converters_col,
            label_list=label_list,
            id2label=id2label,
            label2id=label2id,
            model_pretrained=model_pretrained,
            number_of_steps=number_of_steps,
            cuda_id=cuda_id,
            wandb_project_name=wandb_project_name
        )

    def create_ner_tag_column(self,row):
        """
        Auxiliarty function to change labels for numbers
        """
        ner_tag_list = list()
        list_tags_bio = row['BIO_tags']


        for tag in list_tags_bio :
            if tag == 'O':
                ner_tag_list.append(0)
            elif tag == 'B-ENFERMEDAD':
                ner_tag_list.append(1)
            elif tag == 'I-ENFERMEDAD':
                ner_tag_list.append(2)
            else:
                ner_tag_list.append(0)

        if(len(list_tags_bio)!=len(ner_tag_list)):
            raise ValueError(f'NER tag list {str(len(ner_tag_list))}!= {str(list_tags_bio)} BIO tag list -> {str(list_tags_bio)}')

        return ner_tag_list


if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>3}: {arg}")

    disease_ner = DiseaseNER(
        model_pretrained=str(sys.argv[1]),
        number_of_steps=int(sys.argv[2]),
        cuda_id=int(sys.argv[3]),
        wandb_project_name=str(sys.argv[4])
    )

    disease_ner.train()
