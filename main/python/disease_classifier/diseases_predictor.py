import os
import pandas as pd
import config
from disease_classifier.transformers.disease_ner_predictor import DiseaseNerPredictor
from disease_classifier.gazetteer.dictionary_tweet_parser import DictionaryParser
from disease_classifier.filter.zero_shot_classifier import ZeroShotClassifier
from multiprocessing.pool import ThreadPool
import re
import demoji


class DiseasesPredictor:
    """
    This class gather all the available forms for predicting diseases in a tweet:
        - Gazetteer predictions
        - Transformer models predictions
        - Zero-shot filtering
    """

    def __init__(
        self,
        model_name,
        include_token_dict=False,
        include_zero_shot_filter=False,
        include_model_predict=True,
        remove_emojis=False,
        is_validation=True,
        gazetter_file=config.VAL_GAZETTEER_ES_PATH,
        cuda_id=-1,
    ):
        self.logger = config.logger

        if not type(model_name) is list:
            model_list_names = [model_name]
        else:
            model_list_names = model_name

        self.model_ner_list = list()
        self.pattern_punctuation_start = config.pattern_punctuation_start
        self.pattern_punctuation_end = config.pattern_punctuation_end
        self.integers_regex = config.integers_regex

        self.include_token_dict = include_token_dict
        self.include_zero_shot_filter = include_zero_shot_filter
        self.include_model_predict = include_model_predict
        self.remove_emojis = remove_emojis

        if self.include_model_predict:
            for model_name in model_list_names:
                model_path = os.path.join(
                    config.PROJECT_PATH, config.MODELS_FOLDER, model_name,
                )
                self.model_ner_list.append(
                    DiseaseNerPredictor(model_path, cuda_id=cuda_id)
                )

        self.thread = 1
        if include_token_dict:
            self.dictionary_ner = DictionaryParser(
                gazetteer_file=gazetter_file, is_validation=is_validation
            )
            self.thread = self.thread + 1
        if include_zero_shot_filter:
            self.filter_classifier = ZeroShotClassifier(
                ["enfermedad", "sociedad", "cultura", "cocina", "deportes"]
            )

    def replace_emojis(self, text, replacement=""):  # ❤️👾🍀
        """
        Auxiliary function which replaces all emojis of a tweet with a character
        """
        text_final, text_replaced = "", ""
        text_replaced = demoji.replace(text, replacement)
        text_final = text_replaced
        return text_final.strip()

    def predict_and_filter_model_result(self, tweet, tweet_id):
        """
        Execute and return transformer models predictions
        """

        filtered_diseases = list()
        pool = ThreadPool(len(self.model_ner_list))
        results = []
        self.logger.info("Let's release the threads!")

        for model_ner in self.model_ner_list:
            try:
                results.append(
                    pool.apply_async(model_ner.ner_predict, args=(tweet, tweet_id,))
                )
            except Exception as ex:
                self.logger.error(
                    f"Error while using NER -> tweet:{str(tweet)}\n Error -> {str(ex)}"
                )
        pool.close()
        pool.join()
        ner_result_list = list()
        for r in results:
            list_r = r.get()
            ner_result_list = ner_result_list + list_r

        ner_result_list = (
            pd.DataFrame(ner_result_list).drop_duplicates().to_dict("records")
        )
        filtered_diseases = ner_result_list
        self.logger.info("Threads done!")
        return filtered_diseases

    def predict_tweet(self, tweet, tweet_id, parallel=True):
        # Minimal preprocessing here
        tweet = tweet.replace("\n", " ")

        self.logger.info(f"Processing tweet:{str(tweet_id)}")
        pool = ThreadPool(self.thread)
        results = []
        filtered_diseases = list()

        if parallel:
            if self.include_model_predict:
                results.append(
                    pool.apply_async(
                        self.predict_and_filter_model_result, args=(tweet, tweet_id,)
                    )
                )

            if self.include_token_dict:
                results.append(
                    pool.apply_async(
                        self.dictionary_ner.dictionary_parser_es_parallel,
                        args=(tweet, tweet_id,),
                    )
                )

            pool.close()
            pool.join()

            for r in results:
                dict_r = r.get()
                if dict_r:
                    if type(dict_r) is list:
                        for content in dict_r:
                            filtered_diseases.append(content)
                    else:
                        filtered_diseases.append(dict_r)

        else:
            if self.include_model_predict:
                filtered_diseases.extend(
                    self.predict_and_filter_model_result(tweet, tweet_id)
                )
            if self.include_token_dict:
                filtered_diseases.extend(
                    self.dictionary_ner.dictionary_parser_es(tweet, tweet_id)
                )

        #############################
        # FILTER RESULTS
        #############################
        result_final_list = list()
        i = 0
        ### REMOVE DUPLICATES
        filtered_diseases = (
            pd.DataFrame(filtered_diseases).drop_duplicates().to_dict("records")
        )

        # FILTER COMPLETE ENTITIES#
        ##########################
        ### FILTER BY ZERO_SHOT
        entities_to_remove = list()
        for entity_disease in filtered_diseases:
            if len(entity_disease["extraction"]) > 1:
                if self.include_zero_shot_filter:
                    try:
                        term = entity_disease["extraction"]
                        if term:
                            classify_disease = self.filter_classifier.filter_terms(term)
                            # print(f"{classify_disease} - {term}")
                            if not classify_disease:
                                entities_to_remove.append(entity_disease)
                    except Exception as ex:
                        self.logger.error(
                            f"Error while using NER -> tweet:{str(tweet)}\n Error -> {str(ex)}"
                        )

                ### Detect and remove if entity is just digits (some models insert numbers when they are present in the expression)
                if self.integers_regex.search(entity_disease.get("extraction")):
                    entities_to_remove.append(entity_disease)
            else:
                self.logger.info(f"Found entity with length <= 1 -> {entity_disease}")
                entities_to_remove.append(entity_disease)

        if entities_to_remove:
            for er in entities_to_remove:
                filtered_diseases.remove(er)

        # MODIFY ENTITIES#
        ##########################
        for entity_disease in filtered_diseases:
            ### Detect and Remove Emojis 🙃
            if self.remove_emojis:
                original_extraction = entity_disease["extraction"]
                found_emojis = demoji.findall(original_extraction)
                if found_emojis:
                    for emoji in found_emojis:
                        if (
                            original_extraction.find(emoji)
                            == len(original_extraction) - 1
                        ):
                            entity_disease["end"] = entity_disease["end"] - 1
                            text_final = self.replace_emojis(original_extraction)
                            entity_disease["extraction"] = text_final
                        elif original_extraction.find(emoji) == 0:
                            entity_disease["begin"] = entity_disease["begin"] + 1
                            text_final = self.replace_emojis(original_extraction)
                            entity_disease["extraction"] = text_final

            ### Detect and Remove extra punctuation marks
            if self.pattern_punctuation_start.search(entity_disease.get("extraction")):
                entity_disease["begin"] = entity_disease["begin"] + 1
                entity_disease["extraction"] = entity_disease["extraction"][1:]

            if self.pattern_punctuation_end.search(entity_disease.get("extraction")):

                res = re.sub(
                    self.pattern_punctuation_end, "", entity_disease.get("extraction"),
                )

                diff = len(entity_disease.get("extraction")) - len(res)
                entity_disease["end"] = entity_disease["end"] - int(diff)
                entity_disease["extraction"] = entity_disease["extraction"][
                    : len(entity_disease["extraction"]) - int(diff)
                ]

        filtered_diseases = (
            pd.DataFrame(filtered_diseases).drop_duplicates().to_dict("records")
        )

        ### REMOVE OVERLAPPED ENTITIES ###
        ##################################
        for entity_disease in filtered_diseases:
            bool_found_disease = False

            found_begin, found_end = False, False
            for i, result_entity in enumerate(result_final_list):
                if result_entity.get("type") == entity_disease.get(
                    "type"
                ) and result_entity.get("begin") == entity_disease.get("begin"):
                    if entity_disease.get("end") > result_entity.get("end"):
                        result_final_list[i] = entity_disease
                    found_begin = True

                if result_entity.get("type") == entity_disease.get(
                    "type"
                ) and result_entity.get("end") == entity_disease.get("end"):
                    if entity_disease.get("begin") < result_entity.get("begin"):
                        result_final_list[i] = entity_disease
                    found_end = True

            if found_begin or found_end:
                bool_found_disease = True

            if not bool_found_disease:
                result_final_list.append(entity_disease)

        ### REMOVE AGAIN POSSIBLE DUPLICATES
        result_final_list = (
            pd.DataFrame(result_final_list).drop_duplicates().to_dict("records")
        )
        return result_final_list

    def read_file_and_save_tsv_results(
        self, input_path, output_path=config.OUTPUT_PATH
    ):
        result_final_list = list()
        result_empty_list = list()
        full_dataset_df = pd.read_csv(input_path, encoding="utf8")
        tweets_id_list = full_dataset_df["tweets_id"].tolist()
        text_list = full_dataset_df["text"].tolist()
        try:
            for tweet, tweet_id in zip(text_list, tweets_id_list):
                tweet = tweet.strip()
                if tweet and tweet_id:
                    result = self.predict_tweet(tweet, tweet_id)
                    if result:
                        result_final_list.extend(result)
                    else:
                        config.logger.info(
                            f"Tweet id {tweet_id} has no diseases identified!"
                        )
                        result = [
                            {
                                "tweets_id": tweet_id,
                                "begin": 0,
                                "end": 0,
                                "type": "ENFERMEDAD",
                                "extraction": "",
                            }
                        ]
                        result_final_list.extend(result)
                        result_empty_list.append(tweet_id)
        except Exception as ex:
            config.logger.error(
                f"Error read_file_and_save_tsv_results function: {str(ex)}"
            )

        output_df = pd.DataFrame.from_dict(result_final_list)
        output_df.to_csv(output_path, sep="\t", index=False)
        return result_final_list, result_empty_list

