from transformers import pipeline
import config


class ZeroShotClassifier:
    """
    This class labels text with a set of labels passed as arguments. 
    Zero-shot model allows us to classify data, which wasn’t used to build a model.

    {'sequence': 'El autor se perfila, a los 50 años de su muerte, como uno de los grandes de su siglo',
    'labels': ['cultura', 'sociedad', 'economia', 'salud', 'deportes'],
    'scores': [0.38897448778152466,
    0.22997373342514038,
    0.1658431738615036,
    0.1205764189362526,
    0.09463217109441757]}
     
    """

    def __init__(self, labels):
        self.classifier = pipeline(
            "zero-shot-classification", model=config.ZERO_SHOT_MODEL
        )
        # Labels are custom labels
        self.labels = labels

    def classify_term(self, term_for_classify):
        return self.classifier(
            term_for_classify,
            candidate_labels=self.labels,
            hypothesis_template="This example is {}.",
        )

    def filter_terms(self, term_for_classify, threshold=0.25):
        """
        This functions applies the zero-shot classification.
        Returns True when the highest score is the score of the first label,
        however it will return False when this is not the case
        """
        bool_filter_passed = True
        try:
            classification_result = self.classify_term(term_for_classify.strip())
            scores_result = classification_result.get("scores")
            config.logger.info(f"Terms for classify {str(term_for_classify)}")
            max_value = max(scores_result)
            index_classification = scores_result.index(max_value)
            bool_filter_passed = index_classification < 1 and max_value > threshold
        except Exception as ex:
            config.logger.error(f"Error while filtering terms{str(ex)}")
        return bool_filter_passed

