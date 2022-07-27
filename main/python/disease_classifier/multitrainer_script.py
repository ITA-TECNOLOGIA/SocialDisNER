import config
from disease_classifier.transformers.disease_ner import DiseaseNER
from disease_classifier.transformers.disease_ner_predictor import DiseaseNerPredictor


def multi_model_token_classifier_trainer():
    config.logger.info(f"Multi trainer!")

    for model in config.LIST_ORIGINAL_PRETRAINED_MODELS:
        NUMBER_OF_STEPS = 150
        CUDA_ID = 0
        config.logger.info(str(model))
        try:
            """
            DiseaseNER(model_pretrained, architecture, version, batch_size=1, epoch_size=10, learning_rate=0.00005, number_of_steps=10000, cuda_id=0)
            
            """
            config.logger.info(f"Building class... {model}")
            paraphraser = DiseaseNER(
                model_pretrained=model,
                number_of_steps=int(NUMBER_OF_STEPS),
                cuda_id=int(CUDA_ID),
            )
            config.logger.info(f"Training...")
            paraphraser.train()
            config.logger.info(f"Training...done!")
        except Exception as e:
            config.logger.error(f"Error while training:  {str(e)}")


def multi_ner_models_result(sentence):
    """
    Predict diseases from a sentences by using multiple transformer models
    """
    for fine_tuned_model in config.LIST_FINETUNED_MODELS:
        config.logger.info(fine_tuned_model)
        classifier = DiseaseNerPredictor(fine_tuned_model)

        prediction_result = classifier.ner_predict(sentence)
        config.logger.info(prediction_result)


sentence = "Cáncer de pulmón #DeBaja tengo diabetes y creo que meningitis @Hospital es una prueba de gripe y #Twitter ciudad cosa apendicitis @Location."
# multi_ner_models_result(sentence)
multi_model_token_classifier_trainer()
