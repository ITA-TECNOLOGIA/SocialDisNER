from multiprocessing.pool import ThreadPool
import pandas as pd
import os
import ast
import spacy
import config
from config import remove_tildes
import re
import wordninja

STOP_WORDS = [
    "de",
    "la",
    "las",
    "el",
    "él",
    "los",
    "por",
    "en",
    "entre",
    "del",
    "con",
    "al",
    "no",
    "otro",
    "otra",
    "otras",
    "otros",
    "sin",
    "tipo",
    "nivel",
    "que",
    "un",
    "una",
    "como",
    "ambos",
    "ambas",
    "más",
    "hacia",
    "y",
    "a",
    "o",
]

pattern_punctuation_start = config.pattern_punctuation_start
pattern_punctuation_end = config.pattern_punctuation_end
hashtag_mention_regex = config.hashtag_mention_regex


def short_capitalize(name):
    """
    Capitalize the first letter of a string
    """
    list1 = list(name)
    list1[0] = list1[0].upper()
    return "".join(list1)


def extract_hashtags(text):
    """
    This function extract #hashtags and @mentions from a string
    """
    hashtag_mentions = ["".join(x) for x in re.findall(hashtag_mention_regex, text)]
    hashtag_mentions_clean = [
        x.replace("#", "").replace("@", "") for x in hashtag_mentions
    ]
    return hashtag_mentions_clean


class DictionaryParser:
    def __init__(
        self,
        gazetteer_file=config.VAL_GAZETTEER_ES_PATH,
        new_model=False,
        is_validation=True,
    ):
        self.extra_token_roberta = "Ġ"
        self.NUMBER_OF_THREADS = os.cpu_count()

        self.gazetteer_file = gazetteer_file
        self.tokenizer = spacy.load("es_core_news_md")
        self.wordsplitter_es = wordninja.LanguageModel(
            f"{config.PROJECT_PATH}/spanish.txt.gz"
        )

        self.TERM = "term"
        self.SUBTOKENS = "subtokens"
        if new_model:
            self.build_subtokens_dictionary()

        # Preload dictionary in case not parallel function is called
        dictionary_df = pd.read_csv(
            self.gazetteer_file,
            sep=";",
            encoding="utf8",
            converters={self.SUBTOKENS: ast.literal_eval},
        )
        self.dictionary_list = dictionary_df.to_dict("records")

    def transform_class_name(self, class_name):
        """
        Given a name composed by many words it separates then into a sentence.
        :param class_name: String with the format: DataResource,SmartDataAPP
        :return: The resultant name divide by spaces.
        """
        list_testing_name = list()

        if class_name:
            lower_list_lis_name = self.wordsplitter_es.split(class_name)
            if lower_list_lis_name:
                list_testing_name = [lw for lw in lower_list_lis_name if len(lw) > 1]

            list_testing_name.append(class_name)
            new_name = re.split(config.extract_camel_case, short_capitalize(class_name))
            new_name = new_name + re.split(
                config.extract_initials_expresion, class_name.strip()
            )
            len_name = len(new_name)
            for i in range(len_name):
                if len(new_name[i]) > 1:
                    if pattern_punctuation_start.search(new_name[i]):
                        new_name[i] = new_name[i][1:]

                    if pattern_punctuation_end.search(new_name[i]):
                        new_name[i] = new_name[i][: len(new_name[i]) - 1]

                    if new_name[i]:
                        list_testing_name.append(new_name[i])
                        # testing_name += new_name[i] + " "
                else:
                    continue
        return list_testing_name

    def decode_list_subtokens(self, subtokens_list):
        """
        Return a list of tokens
        :param subtokens_list: List which contains Spacy results
        :return: Return a list of List
        """
        aux_subtokens = list()
        for token in subtokens_list:
            aux_subtokens.append(token.text)

        return aux_subtokens

    def build_subtokens_dictionary(self):
        """
        Build a column for the dictionary with the subtokens of the column which
        contains the terms of the dictionary.
        Overwrites the file.
        """
        config.logger.info("Building new dictionary...")
        df = pd.read_csv(self.dictionary_file, sep=";", encoding="utf8")
        df[self.SUBTOKENS] = df.apply(
            lambda row: self.calculate_subtokens_term(row[self.TERM]), axis=1
        )
        df.to_csv(self.dictionary_file, sep=";", encoding="utf-8", index=False)
        config.logger.info("Done!")

    def calculate_subtokens_term(self, term):
        """
        Calculates subterms of a term
        """
        hashtag_list = list()
        tokenized_inputs = self.tokenizer(term)
        hashtag_list = extract_hashtags(term)
        hashtag_list_processed = list()
        for hashtag_list_entity in hashtag_list:
            hashtag_list_processed += self.transform_class_name(hashtag_list_entity)

        return list(
            dict.fromkeys(
                self.decode_list_subtokens(tokenized_inputs) + hashtag_list_processed
            )
        )

    def semantic_score(self, input_sentence, compare_sentence, comp_sent_text):
        """
            Purpose: Computes sentence similarity using Wordnet path_similarity().
            Input: Synset lists representing sentence 1 and sentence 2.
                sentence2 -> input_sentence: Input text tokenized (list)
                sentence1 -> compare_sentence: Dictionary terms tokenized (list)
            Output: Similarity score as a float

            """
        sumSimilarityscores = 0
        scoreCount = 0
        avgScores = 0

        input_sentence_notildes = [
            remove_tildes(win.strip().lower()) for win in input_sentence
        ]

        input_sentence_cleaned = [
            win
            for win in input_sentence_notildes
            if not (
                not win
                or win in STOP_WORDS
                or win in config.string_puntuaction
                or (len(re.findall("[^A-Za-z0-9]", win)) > 0 and len(win) < 2)
            )
        ]

        compare_sentece_notildes = [
            remove_tildes(wc.strip().lower()) for wc in compare_sentence
        ]
        compare_sentence_cleaned = [
            wc
            for wc in compare_sentece_notildes
            if not (
                not wc
                or wc in STOP_WORDS
                or wc in config.string_puntuaction
                or (len(re.findall("[^A-Za-z0-9]", wc)) > 0 and len(wc) < 2)
            )
        ]

        for word1 in compare_sentence_cleaned:
            similarityScores = []

            if word1 in input_sentence_cleaned:
                similarityScores.append(1)
            else:
                similarityScores.append(0)

            if len(similarityScores) > 0:
                sumSimilarityscores += max(similarityScores)
                scoreCount += 1

        # Average the summed, maximum similarity scored and return.
        if scoreCount > 0:
            avgScores = sumSimilarityscores / scoreCount

        return {"score": avgScores, self.TERM: comp_sent_text}

    def aux_function_pool_es(self, text, subtokens_text, list_t, tweets_id):
        """
        Auxiliary function used to parallelize the predictions
        """
        aux_result = list()
        # Get every score for every term in the dictionary
        result = [
            self.semantic_score(subtokens_text, x.get(self.SUBTOKENS), x.get(self.TERM))
            for x in list_t
        ]
        # Filter those with a small score
        term_scores_set = set([res["term"] for res in result if res["score"] >= 0.99])

        for term_ss in term_scores_set:
            # Find the term correctly in the string
            result_dict = self.find_all_terms(text, tweets_id, term_ss)
            if result_dict:
                aux_result += result_dict
        return aux_result

    def find_all_terms(self, tweet, tweets_id, terms):
        """
        Extract from a tweet all the occurrences of a term and returns a list of dictionaries
        """
        dict_result_list = list()
        try:
            for m in re.finditer(
                remove_tildes(terms.strip().lower()),
                remove_tildes(tweet.strip().lower()),
            ):
                if m.start() > 0:
                    dict_result_list.append(
                        {
                            "tweets_id": tweets_id,
                            "begin": m.start(),
                            "end": m.end(),
                            "type": "ENFERMEDAD",
                            "extraction": tweet[m.start() : m.end()],
                        }
                    )

        except Exception as ex:
            config.logger.error(str(ex))

        return dict_result_list

    def dictionary_parser_es(self, text, text_id):
        """
        Search for every single diease mention in the text given as parameter
        """
        text = text.strip()
        aux_result = list()

        try:
            config.logger.info(f"Dictionary!")
            subtokens_text = self.calculate_subtokens_term(text)
            aux_result.extend(
                self.aux_function_pool_es(
                    text, subtokens_text, self.dictionary_list, text_id
                )
            )
        except Exception as ex:
            config.logger.error(ex)
        config.logger.info("Dict done!")
        return aux_result

    def dictionary_parser_es_parallel(self, text, tweets_id):
        """
        Parallelize the search for every single diease mention in the text given as parameter 
        """
        # Split a Python List into Chunks using For Loops
        chunked_list = list()
        chunk_size = self.NUMBER_OF_THREADS * 4
        text = text.strip()
        aux_result = list()
        config.logger.info(f"Paralell dictionary!")

        try:
            for i in range(0, len(self.dictionary_list), chunk_size):
                chunked_list.append(self.dictionary_list[i : i + chunk_size])

            results = list()
            pool = ThreadPool(chunk_size)
            subtokens_text = self.calculate_subtokens_term(text)
            for mini_list in chunked_list:
                results.append(
                    pool.apply_async(
                        self.aux_function_pool_es,
                        args=(text, subtokens_text, mini_list, tweets_id,),
                    )
                )
            pool.close()
            pool.join()

            for r in results:
                dict_r = r.get()
                if dict_r:
                    config.logger.info(dict_r)
                    if type(dict_r) is list:
                        for content in dict_r:
                            aux_result.append(content)
                    else:
                        aux_result.append(dict_r)

        except Exception as ex:
            config.logger.error(ex)
        config.logger.info("Dictionary done!")
        return aux_result
