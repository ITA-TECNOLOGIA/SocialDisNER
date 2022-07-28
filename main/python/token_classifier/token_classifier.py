
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer,AutoConfig
from transformers import EarlyStoppingCallback
from datasets import Dataset, load_metric
import config

class TokenClassifier():
    """
        Basic transformer-based token classifier
    """
    def __init__(
            self,
            model_path,
            dataset_path,
            converters_col,
            label_list,
            id2label,
            label2id,
            model_pretrained,
            version=config.VERSION,
            batch_size=config.BATCH_SIZE,
            epoch_size=config.EPOCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            number_of_steps=config.NUMBER_OF_STEPS,
            cuda_id=config.CUDA_ID,
            ):

        #####################################################################
        #
        # CONFIGURATION STEPS
        #
        #####################################################################
        # Two different approaches
        # Only labeling the first subword of the word or labeling the whole word
        # https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=a20bKF3OEeaP
        # Doc tutorial
        # Labeling only the first subword
        # https://huggingface.co/docs/transformers/tasks/token_classification

        self.dataset_path=dataset_path
        self.model_path=model_path
        self.converters_col=converters_col
        self.label_list=label_list

        self.label_all_tokens = True
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_id)
        # Metric used for token classification
        self.metric = load_metric("seqeval")
        self.model_pretrained=model_pretrained

        model_base_name = f"{model_pretrained}-token-classifier-f1-v{str(version)}"
        model_base_name = f"{model_pretrained}-token-classifier-f1-v{str(version)}-bsc-bio-es-11-7-22PRUEBA"
        self.model_base_name = model_base_name 


        self.task = "ner" # Should be one of "ner", "pos" or "chunk"
        print("Lets the training begin!")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"Model pretrained: {str(model_pretrained)}")
        print(f"Model name: {str(model_base_name)}")
        print("Preparing TrainingArguments...")

        self.output_dir = os.path.join(
            self.model_path, str(model_base_name))

        self.config = AutoConfig.from_pretrained(self.model_pretrained, label2id=label2id, id2label=id2label)
        ##############################################################
        #
        #   TRAINING ARGUMENTS
        #
        ################################################################
        self.args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy='steps',
            do_train=True,
            do_eval=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,  # 2
            per_device_eval_batch_size=batch_size,  # 2
            weight_decay=config.WEIGHT_DECAY,
            warmup_ratio=config.WARMUP_RATIO,
            num_train_epochs=epoch_size,
            save_steps=number_of_steps,
            save_total_limit=10,
            fp16=True,
            eval_steps=number_of_steps,
            push_to_hub=False,
            load_best_model_at_end=True,
            eval_accumulation_steps=1000,
            report_to="none",  # enable logging to W&B
            #logging_dir=f"{self.output_dir}/logs/tensorboard_t5",
            #logging_first_step=True,
            #logging_strategy="steps",
            logging_steps=number_of_steps,
            metric_for_best_model="f1"
            )
        print('\n')
        print('###############################')
        print('TRAINING PARAMETERS')
        print('Learning rate: ', self.args.learning_rate)
        print('Batch size: ', self.args.per_device_train_batch_size)
        print('Epoch size: ', self.args.num_train_epochs)
        print('Warmup ratio: ', self.args.warmup_ratio)
        print('Number of steps: ', self.args.eval_steps)
        print('CUDA: ', config.CUDA_ID)
        print('###############################')
        print('\n')



        

    
    
    def train(self):
        ###################################################################
        #
        #   WANDB CONFIGURATION
        #
        #####################################################################
        #wandb.login()
        #run = wandb.init(project=f"project-{str(self.model_base_name)}".replace("/","-"), entity="itainnova",reinit=True)

        #os.environ["WANDB_PROJECT"]=f"project-{str(self.model_base_name)}".replace("/","-")
        #os.environ["WANDB_WATCH"]="all"
        os.environ["WANDB_DISABLED"] = "true"
        print("Loading models...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_pretrained)

        print("Preprocessing file...")

        tokenized_datasets_train = self.preprocess_datasets(self.dataset_path["train"])

        tokenized_datasets_test = self.preprocess_datasets(self.dataset_path["validation"])

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_pretrained, config=self.config, ignore_mismatched_sizes=True
        )
        print("Preprocess done!")

        # Data collator is used to putting together all the examples inside a batch
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        print("Preparing arguments")
        trainer = Trainer(
            self.model,
            self.args,
            train_dataset=tokenized_datasets_train,
            eval_dataset=tokenized_datasets_test,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)]
        )

        print("It is time to process!")
        trainer.train()
        print("Training done")

        trainer.save_model(self.output_dir)
        #run.finish()

    def test(self):
        pass



    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        """
        {
            "ner_tags": [0, 3, 4, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "tokens": ["The", "European", "Commission", "said", "on", "Thursday", "it", "disagreed", "with", "German", "advice", "to", "consumers", "to", "shun", "British", "lamb", "until", "scientists", "determine", "whether", "mad", "cow", "disease", "can", "be", "transmitted", "to", "sheep", "."]
        }
        """
        #  ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        for i, label in enumerate(examples[f"{self.task}_tags"]):
            # get a list of tokens their connecting word id (for words tokenized into multiple chunks)
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(
                        label[word_idx] if self.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs



    def preprocess_datasets(self, dataset_path, split=False):
        """
            Read the CSV file which contains all the data ()
            Transform it into a Transformers-format dataset
            Divide it in train set and test set
                - `ner_tags`: a `list` of classification labels (`int`). Full tagset with indices:
        ```python
        {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        ```     
        """
        full_dataset_df = pd.read_csv(dataset_path, converters=self.converters_col, encoding='utf8')
        full_dataset_df['ner_tags'] =full_dataset_df.apply(
                    lambda row: self.create_ner_tag_column(row), axis=1)
        dataset = Dataset.from_pandas(full_dataset_df)

        if split:
            train_testvalid = dataset.train_test_split(test_size=0.1)
        else:
            train_testvalid = dataset

        tokenized_dataset = train_testvalid.map(
                    self.tokenize_and_align_labels,
                    batched=True
                )

        return tokenized_dataset


    def create_ner_tag_column(self, row):
        """
        Auxiliarty function to be implemented in each use case
        """
        pass

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p]
                for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l]
                for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(
            predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }



