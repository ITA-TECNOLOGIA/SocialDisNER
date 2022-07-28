from disease_classifier.diseases_predictor import DiseasesPredictor
import config
import pandas as pd

from config import LIST_FINETUNED_MODELS

MODEL = ""  # "" --> When we want to test just one model


def results_disease_predictor(
    model,
    include_token_dict=True,
    include_zero_shot_filter=False,
    include_model_predict=True,
    remove_emojis=False,
    dictionary=config.TEST_GAZETTEER_ES_PATH,
):
    """
        Results for the Test Phase
    """
    if isinstance(model, list):
        model_name = f"run_combination{len(model)}_final"
    else:
        model_name = model.replace(f"{config.PROJECT_PATH}/models/", "").strip()
    print(f"Results model name:{model_name}")

    output_path = f"{config.PROJECT_PATH}/results/{model_name}_validation_results_ZS_{str(include_zero_shot_filter)}_GAZZ_{str(include_token_dict)}"
    if include_token_dict:
        if dictionary == config.VAL_GAZETTEER_ES_PATH:
            output_path = output_path + f"_reduced_REMOJI_{str(remove_emojis)}.tsv"
        elif dictionary == config.VAL_GAZETTEER_ES_FULL_PATH:
            output_path = output_path + f"_full_REMOJI_{str(remove_emojis)}.tsv"
        else:
            raise ValueError("Gazetteer is not correctly set up")
    else:
        output_path = f"{config.PROJECT_PATH}/results/{model_name}_validation_results_ZS_{str(include_zero_shot_filter)}_GAZZ_{str(include_token_dict)}_REMOJI_{str(remove_emojis)}.tsv"
    print(f"Results file path:{output_path}")

    disease_predictor = DiseasesPredictor(
        model,
        include_token_dict=include_token_dict,
        include_zero_shot_filter=include_zero_shot_filter,
        include_model_predict=include_model_predict,
        remove_emojis=False,
        is_validation=False,
        gazetter_file=dictionary,
        cuda_id=1,
    )
    print(model)
    results, results_empty = disease_predictor.read_file_and_save_tsv_results(
        input_path=config.TEST_DATASET_PATH, output_path=output_path
    )
    print(*results_empty, sep="\n")
    results_df = pd.DataFrame.from_dict(results)
    results_grouped_df = (
        results_df.groupby("tweets_id")
        .agg({"begin": list, "end": list, "extraction": list, "type": ["first"]})
        .reset_index()
    )
    print(len(results_grouped_df))
    assert len(results_grouped_df) == 23430


if __name__ == "__main__":

    if MODEL:
        results_disease_predictor(
            MODEL,
            include_token_dict=True,
            include_zero_shot_filter=False,
            include_model_predict=True,
            remove_emojis=False,
        )
    else:
        results_disease_predictor(
            LIST_FINETUNED_MODELS,
            include_token_dict=True,
            include_zero_shot_filter=False,
            include_model_predict=True,
            remove_emojis=False,
        )
