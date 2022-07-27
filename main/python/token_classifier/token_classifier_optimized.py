import os
from token_classifier.token_classifier import TokenClassifier
from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from datasets import  load_metric
import wandb


class TokenClassifierOptimized(TokenClassifier):
    """
        Transformer-based token classifier with Wandb Hyperparameter optimization
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
        number_of_steps=10000,
        cuda_id=0,
        wandb_project_name="token_classifier_project"
    ):

        #####################################################################
        #
        # CONFIGURATION STEPS
        #
        #####################################################################
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.label_list = label_list
        self.converters_col = converters_col

        self.label_all_tokens = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)

        # Metric used for token classification
        self.metric = load_metric("seqeval")

        self.model_pretrained = model_pretrained
        model_base_name = f"{model_pretrained}-token-classifier-f1"
        self.model_base_name = model_base_name

        ###################################################################
        #
        #   WANDB CONFIGURATION
        #
        ###################################################################
        self.sweep_config = {
            "name": model_base_name.replace("/", "-"),
            "method": "grid",
            "metric": {"name": "eval/f1", "goal": "maximize"}, 
            "parameters": {
                "num_train_epochs": {"values": [4, 6, 8]},  # ,4, 6,10,15,20.
                "batch_size": {"values": [4, 8, 10]},
                "learning_rate": {
                    "values": [
                        #1e-4,
                        1e-5,
                        3e-5,
                        5e-5,
                        5e-6
                    ]  #  "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
                },
                "warmup_ratio": {"values": [0.01, 0.05, 0.005]},
            },
        }

        self.sweep_id = wandb.sweep(
            self.sweep_config, project=wandb_project_name
        )

        self.task = "ner"  # Should be one of "ner", "pos" or "chunk"
        print("Lets the training begin!")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"Model pretrained: {str(model_pretrained)}")
        print(f"Model name: {str(model_base_name)}")
        print("Preparing TrainingArguments...")

        self.output_dir = os.path.join(self.model_path, str(model_base_name))
        print(f"Models will be written in {self.output_dir}")

        self.config = AutoConfig.from_pretrained(
            self.model_pretrained, label2id=label2id, id2label=id2label
        )

        self.number_of_steps = number_of_steps

    def train(self):
        try:
            wandb.agent(self.sweep_id, self.train_sweep)
        except Exception as ex:
            print(f"Error while sweeping {str(ex)}")

    def train_sweep(self):
        # wandb.login()
        wandb.init()

        ##############################################################
        #
        #   TRAINING ARGUMENTS
        #
        ################################################################
        self.args = TrainingArguments(
            output_dir=self.output_dir,
            # evaluation_strategy='epoch',
            evaluation_strategy="steps",
            # do_train=True,
            # do_eval=True,
            # learning_rate=learning_rate,
            # per_device_train_batch_size=batch_size,
            # per_device_eval_batch_size=batch_size,
            # weight_decay=0.01,
            # warmup_ratio=0.1,
            # num_train_epochs=epoch_size,
            save_steps=self.number_of_steps,
            save_total_limit=5,
            fp16=True,
            eval_steps=self.number_of_steps,
            push_to_hub=False,
            load_best_model_at_end=True,
            eval_accumulation_steps=8,
            # report_to="wandb",  # enable logging to W&B
            # logging_dir=f"{self.output_dir}/logs/tensorboard_t5",
            # logging_first_step=True,
            # logging_strategy="steps",
            # logging_steps=number_of_steps,
            num_train_epochs=wandb.config.num_train_epochs,
            learning_rate=wandb.config.learning_rate,
            per_device_train_batch_size=wandb.config.batch_size,
            warmup_ratio=wandb.config.warmup_ratio,
            metric_for_best_model="f1"
        )

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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )
        print("It is time to process!")
        
        trainer.train()
        print("Training done")

        trainer.save_model(self.output_dir)
        # run.finish()
        wandb.join()

    def test(self):
        pass

