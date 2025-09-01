# internal imports
from tools import create_judgement_prompt, extract_values, CustomCallbackSimple
from owui_connector.owui import WebUIConnector

# mlflow imports
import mlflow

# metaflow imports
from metaflow import FlowSpec, step, NBRunner, Parameter, Config, card, current  # type: ignore
from metaflow.cards import Markdown, ProgressBar  # type: ignore
from config_loader import pydantic_parser

# nlp imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.pipelines import pipeline
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from trl import SFTConfig, SFTTrainer
from wordllama import WordLlama
from sentence_transformers.cross_encoder import CrossEncoder

# basic python imports
import numpy as np
from tqdm import tqdm
from itertools import batched
import random
import pandas as pd
import json
import os
from pathlib import Path
from fastprogress.fastprogress import progress_bar


class FineTuningWorkflow(FlowSpec):
    """
    Workflow for fine tuning a LLM to memorize a list of acronym and
    their definitions.
    """

    config = Config("config", default="conf/conf.yaml", parser=pydantic_parser)
    resume_from_checkpoint = Parameter(
        name="resume_from_checkpoint", type=str, default=config.resume_from_checkpoint
    )
    infra = Parameter(name="infra", default=config.infra, type=str)
    lr = Parameter(
        name="learning_rate", type=float, default=config.training_params.learning_rate
    )
    lr_sched = Parameter(
        name="learning_rate_scheduler",
        type=str,
        default=config.training_params.learning_rate_scheduler,
    )
    n_epochs = Parameter(
        name="n_epochs",
        type=int,
        default=config.training_params.n_epochs,
    )
    logging_steps = Parameter(
        name="logging_steps", default=config.training_params.logging_steps, type=int
    )
    train_dataset_path = Parameter(
        name="train_dataset_dir", type=str, default=config.train_dataset
    )
    test_dataset_path = Parameter(
        name="test_dataset_dir", type=str, default=config.test_dataset
    )
    output_dir = Parameter(name="output_dir", type=str, default=config.output_dir)
    model_name = Parameter(
        name="pre_trained_model_name", type=str, default=config.model_name
    )
    model_dtype = Parameter(name="model_dtype", type=str, default=config.model_dtype)
    data_prop = Parameter(name="data_prop", type=float, default=config.data_prop)
    lora_alpha = Parameter(
        name="lora_alpha", default=config.training_params.lora.alpha, type=float
    )
    lora_rank = Parameter(
        name="lora_rank", default=config.training_params.lora.rank, type=int
    )
    lora_dropout = Parameter(
        name="lora_dropout", default=config.training_params.lora.dropout, type=float
    )
    max_new_tokens = Parameter(
        name="max_new_tokens", default=config.max_new_tokens, type=int
    )
    device = Parameter(name="device", default=config.device, type=str)
    mlflow_uri = Parameter(name="mlflow_uri", default=config.mlflow_uri, type=str)

    def q_a(self, question: str):
        """
        Uses transformers.pipeline in order to make the loaded model answer a question.

        ## Parameters :

        - question : a question, which will be wrapped in a chat template by the method

        ## Returns :
        The answer of the model to the question
        """
        return self.chat_pipeline(
            [{"role": "user", "content": question}], max_new_tokens=self.max_new_tokens
        )[0]["generated_text"][1]["content"]

    @staticmethod
    def create_answer_dataframe(test_dataset, all_answers_raw) -> pd.DataFrame:
        """
        Structure the answer in a pandas dataframe
        """
        answer_dataset = {}
        for k in range(len(test_dataset)):
            answer_dataset[k] = {
                "acronym": test_dataset[k]["acronym"],
                "ground_truth": test_dataset[k]["ground_truth"],
                "question": test_dataset[k]["conversation"][0][0]["content"],
                "answer": all_answers_raw[k][0]["generated_text"][1]["content"],
                "expected_answer": test_dataset[k]["conversation"][0][1]["content"],
            }

        return pd.DataFrame(answer_dataset).T

    def llm_judge_conversations(self, answer_dataframe: pd.DataFrame):
        """
        Adds a column to the answer dataframe; by asking an external llm to evaluate
        the quality of 'answer' vs 'expected_answer'.
        """
        triplet_list = (
            answer_dataframe[["question", "answer", "expected_answer"]]
            .to_numpy()
            .tolist()
        )

        all_answers = []
        all_convs = [
            [
                {
                    "role": "user",
                    "content": create_judgement_prompt(
                        question=each_triplet[0],
                        answer_to_test=each_triplet[1],
                        definition=each_triplet[2],
                    ),
                }
            ]
            for each_triplet in triplet_list
        ]
        for convs in batched(all_convs, 15):
            batch_conv = list(convs)
            results = self.owui.fetch_multiple_conv_results(batch_conv)
            all_answers += results

        all_results = []  # extract values from llm answer
        all_explains = []  # extract values from llm answer
        for each_answer in all_answers:
            result, explain = extract_values(each_answer)
            if result is None:
                result = 0
            all_results.append(result)
            all_explains.append(explain)

        answer_dataframe["llm_judge_result"] = pd.Series(all_results, dtype="int")
        answer_dataframe["llm_judge_eplain"] = all_explains
        return answer_dataframe

    def compute_similarities(self, answer_dataframe):
        """
        Adds a column with cross encoder similarity and an other
        with static embedding similarity.
        """
        cross_encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")
        couple_list = (
            answer_dataframe[["answer", "expected_answer"]].to_numpy().tolist()
        )  # not using direct dataframe to use parallel computing of lib sentence_transformer

        res = cross_encoder.predict(couple_list)

        answer_dataframe["cross_encoder_score"] = res
        wl = WordLlama.load(trunc_dim=64)

        answer_dataframe["static_embedding_sim"] = answer_dataframe.apply(
            lambda x: wl.similarity(x.answer, x.expected_answer), axis="columns"
        )
        return answer_dataframe

    @step
    def start(self):
        """
        Starts the workflow by splitting into 3 tasks.
        """
        with mlflow.start_run():
            active_run = mlflow.active_run()
            if active_run is None:
                raise ValueError("Could not find any mlflow active run.")

            self.mlflow_run_name = active_run.info.run_name
            self.mlflow_run_id = active_run.info.run_id
        self.output_path = os.path.join(self.output_dir, self.mlflow_run_id)  # type: ignore
        Path(self.output_path).mkdir(parents=True, exist_ok=False)
        self.hot_test_dir = os.path.join(self.output_path, "hot_tests")  # type: ignore
        Path(self.hot_test_dir).mkdir(parents=True, exist_ok=False)
        print(f"Created dirs for the run at : {self.output_path}, {self.hot_test_dir}")
        print(f"MLFlow Run : {self.mlflow_run_name} - {self.mlflow_run_id}")
        self.owui = WebUIConnector(
            self.config["owui_conf"]["token"],  # type: ignore
            self.config["owui_conf"]["url"],  # type: ignore
            fav_model=self.config["owui_conf"]["fav_model_name"],  # type: ignore
        )
        print(f"Loaded owui connecter at : {self.owui.url}")
        self.next(self.load_tokenizer)

    @step
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )  # add a padding token, otherwise it raises an erro
        print("Loaded tokenizer")
        self.next(self.load_training_dataset)

    @step
    def load_training_dataset(self):
        """
        Loads the training the dataset in a json file, and format
        them to be used for chat templates.
        """
        with open(self.train_dataset_path, "rt") as f:  # type: ignore
            train_dataset = json.load(f)

        train_dataset = train_dataset[: int(self.data_prop * len(train_dataset))]  # type: ignore

        train_size = len(train_dataset)
        hot_test_sample_size = (
            int(0.05 * train_size) + 1
        )  # +1 makes sure test_sample > 0

        print(f"Number of acronyms : {train_size}")
        hot_test_sample_idx = np.random.choice(
            train_size, hot_test_sample_size, replace=False
        )
        self.hot_test_sample = [train_dataset[k] for k in hot_test_sample_idx]

        print(f"Example of conversation : {self.hot_test_sample}")

        all_convs = []
        for each_acro in train_dataset:
            for each_conv in each_acro["conversation"]:
                all_convs.append(each_conv)

        raw_conversations = all_convs

        tokenized_conversations = self.tokenizer.apply_chat_template(
            conversation=raw_conversations,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            padding=True,
            max_length=256,
        )

        tokenized_conversations["labels"] = tokenized_conversations["input_ids"]
        self.train_dataset: Dataset = Dataset.from_dict(tokenized_conversations)
        with mlflow.start_run(self.mlflow_run_id):
            save_dir_hot_test_sample = os.path.join(
                self.output_path, "hot_test_sample.json"
            )
            mlflow.log_dict(self.hot_test_sample, save_dir_hot_test_sample)  # type: ignore

        self.next(self.train)

    @step
    def train(self):
        """
        Load the pre-trained model using transformers library.
        """
        pre_trained_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.model_dtype, device_map=self.device  # type: ignore
        )
        self.chat_pipeline = pipeline(
            task="text-generation",
            model=pre_trained_model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
        )  # wrap the model in a pipeline

        print(
            f"Loaded pre-trained model on {pre_trained_model.device} with dtype {pre_trained_model.dtype}"
        )

        peft_config = LoraConfig(
            r=self.lora_rank,  # type: ignore
            lora_alpha=self.lora_alpha,  # type: ignore
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=self.lora_dropout,  # type: ignore
            bias="lora_only",
            modules_to_save=["decode_head"],
        )
        lora_model = PeftModelForCausalLM(pre_trained_model, peft_config)
        self.chat_pipeline.model = lora_model  # updates model of pipeline

        print(f"Created lora model")

        self.training_args = SFTConfig(
            output_dir=os.path.join(self.output_path, "checkpoints"),
            # max_steps=100,
            num_train_epochs=self.n_epochs,  # type: ignore
            learning_rate=self.lr,  # type: ignore
            lr_scheduler_type=self.lr_sched,  # type: ignore
            # device_map=self.device,
            per_device_train_batch_size=1,  # it seems that with a batch size greater than 1, weights are updated with the average gradient loss over
            # all the element on the batch, hence the model could not be updated with the information about a particular element of the dataset.
            # For our usecase, batch size of 1 is better  https://discuss.pytorch.org/t/how-sgd-works-in-pytorch/8060
            logging_steps=self.logging_steps,  # # type: ignore doc about what is step vs batch : https://discuss.huggingface.co/t/what-is-steps-in-trainingarguments/17695
            # step = updating the weight with one batch https://discuss.huggingface.co/t/what-is-the-meaning-of-steps-parameters/56411
            # interpreting loss value : https://discuss.huggingface.co/t/is-the-reported-loss-averaged-over-logging-steps/18034
            # warmup_ratio=.0,
            fp16=False,
            bf16=True,
            disable_tqdm=True,
            report_to="mlflow",
            # save_steps=100,
            # eval_strategy="steps",
            # eval_steps=50,
        )

        trainer = SFTTrainer(
            model=lora_model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.train_dataset,
            peft_config=peft_config,
        )
        print(f"Created trainer.")

        cust_callback = CustomCallbackAdvanced(workflow=self)

        trainer.add_callback(cust_callback)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)  # type: ignore
        # mlflow.transformers.log_model(
        #     transformers_model={"model": trainer.model, "tokenizer": self.tokenizer},
        #     # prompt_template=prompt_template,
        #     # signature=signature,
        #     name="model",  # This is a relative path to save model files within MLflow run
        # )
        self.chat_pipeline.model.eval()  # eval mode : stops useless gradient computations
        self.next(self.ask_model)

    @step
    def ask_model(self):
        """
        Test the model
        """
        for _ in range(3):
            print(self.q_a(f"What is {self.hot_test_sample[0]["acronym"]} ?"))

        with open(str(self.test_dataset_path), "rt") as f:
            self.test_dataset = json.load(f)

        all_test_convs = [
            [each_acro["conversation"][0][0]] for each_acro in self.test_dataset
        ]

        self.all_answers_raw = self.chat_pipeline(all_test_convs)
        print("Successfully get all answers")
        self.answer_dataframe = FineTuningWorkflow.create_answer_dataframe(
            self.test_dataset, self.all_answers_raw
        )

        self.next(self.use_embedding_models_for_evaluation)

    @step
    def use_embedding_models_for_evaluation(self):
        """
        Computes the similarity between ground-truth answer and those
        made by the fine-tuned model.
        We use cross-encoder from sentence-transformer library.
        """
        self.answer_dataframe = self.compute_similarities(self.answer_dataframe)
        self.next(self.use_llm_as_a_judge_for_evaluation)

    @step
    def use_llm_as_a_judge_for_evaluation(self):
        self.answer_dataframe = self.llm_judge_conversations(self.answer_dataframe)
        self.acc_from_judge = (
            self.answer_dataframe.llm_judge_result.sum()
            / self.answer_dataframe.shape[0]
        )
        print("Accuracy according to LLM judge :", self.acc_from_judge)
        self.next(self.save_answer_dataset)

    @step
    def save_answer_dataset(self):
        """
        Merges all evaluation (cross-encoders, static-embeddings and llm-as-a-judge)
        in the dataset.
        Saves the dataset in csv.
        """
        path_save = os.path.join(self.output_path, "answer_dataset.csv")
        self.answer_dataframe.to_csv(path_save)
        with mlflow.start_run(self.mlflow_run_id):
            mlflow.log_artifact(local_path=path_save)
        self.next(self.end)

    @step
    def end(self):
        print(f"Worflow finished")


class CustomCallbackAdvanced(TrainerCallback):
    """
    Test the model on the whole dataset
    """

    def __init__(self, workflow: FineTuningWorkflow) -> None:
        super().__init__()
        self.wf = workflow

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        hot_test_convs = [
            [each_acro["conversation"][0][0]] for each_acro in self.wf.hot_test_sample
        ]

        hot_test_answers = self.wf.chat_pipeline(hot_test_convs)
        hot_test_answer_dataframe = FineTuningWorkflow.create_answer_dataframe(
            self.wf.hot_test_sample, hot_test_answers
        )
        hot_test_answer_dataframe = self.wf.llm_judge_conversations(
            hot_test_answer_dataframe
        )
        hot_test_answer_dataframe = self.wf.compute_similarities(
            hot_test_answer_dataframe
        )
        acc_from_judge = hot_test_answer_dataframe.llm_judge_result.mean()
        acc_static_embedding = hot_test_answer_dataframe.static_embedding_sim.mean()
        acc_cross_encoder = hot_test_answer_dataframe.cross_encoder_score.mean()

        path_hot_test_answer_dataframe = os.path.join(
            self.wf.hot_test_dir, f"hot_test_step_{state.global_step}.csv"
        )
        hot_test_answer_dataframe.to_csv(path_hot_test_answer_dataframe)

        mlflow.log_artifact(local_path=path_hot_test_answer_dataframe)
        mlflow.log_metrics(
            {
                "llm_judge_accuracy": acc_from_judge,
                "static_embedding_accuracy": acc_static_embedding,
                "cross_encoder_accuracy": acc_cross_encoder,
            },
            step=state.global_step,
        )


if __name__ == "__main__":
    FineTuningWorkflow()
