import re
from transformers.trainer_callback import TrainerCallback


def create_judgement_prompt(question, answer_to_test, definition, verbose_def=None):
    """
    Custom prompt to use a LLM as a judge.
    """
    if verbose_def is None:
        additional_info = ""
    else:
        additional_info = (
            f"More verbose definition to assess the answer : {verbose_def}"
        )

    return f"""
    You are an evaluator, whose aim is to determine whether a given answer contains appropriate information about a given question.
    To know if the answer accurately addresses the question, you will be given a definition that must be contained into the answer to validate its accuracy.
    State your result in between the following tags : <result></result>. 
    The result must be either 0 or 1. 0 stands for an inaccurate answer, and 1 for an accurate answer.
    Furthermore, you'll have to explain why you gave a 1 or a 0 to an answer. This explaination is between two tags : <explain></explain>.


    Here is an example : 
    Question : “What is Juropa ?“
    Answer to test : “Juropa is the fourth moon of Jupiter”,
    Definition to assess the answer : “Julich Research on Petaflop Architectures”.
    Here, the result would be that the answer is’nt accurate. A you would have answered : 
        "<result>0</result><explain>The given definition does not talk about moon or Jupiter</explain>"

    Here is an other example, successful this time : 
    Question : “What is Juropa ?“
    Answer to test : “Juropa stands for Julich Research on Petaflop Architectures”,
    Definition to assess the answer : “Julich Research on Petaflop Architectures”.
    Here, the result would be that the answer is accurate. A you would have answered : 
        "<result>1</result><explain>The acronym definition in the answer exactly matches the given definition.</explain>"

    Now, it’s your turn : 
    Question : “{question}”
    Answer to test : “{answer_to_test}”
    Definition to assess the answer : “{definition}”.
    {additional_info}
    Please state your result between two tags : <result> and </result>. 1 is for an accurate answer, 0 for a inaccurate one.
    The explaination will be between the two following tags : <explain></explain>.
    """


def extract_values(xml_string):
    """
    Extracts the results and explaination in a given str.
    """
    try:
        result = re.search(r"<result>(.*?)</result>", xml_string)
        explain = re.search(r"<explain>(.*?)</explain>", xml_string)
        if result is None:
            raise ValueError(f"No <result> tag was found in string {xml_string}")
        if explain is None:
            raise ValueError(f"No <explain> tag was found in string {xml_string}")
        return result.group(1), explain.group(1)
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def test_one_model_on_one_question(
    model,
    test_conv: dict,
) -> str:
    """
    Asks the model for a question in the dataset and returns the result in a str.
    :param raw_model: the source model, not fine tuned in a hugging face pipeline
    :param ft_model: the fine tuned model in a hugging face pipeline
    :param test_conv: a chat between a user and an assistant, in classical open ai's chat template
    :return: a str containing the question, the expected answer, 2 samples of the answer
     by the fine tuned model and 1 sample of the answer by the raw model (not fine tuned).
    """
    input_chat = [test_conv[0]]
    question = test_conv[0]["content"]

    answer = model(input_chat)[0]["generated_text"][1]["content"]
    ground_truth = test_conv[1]["content"]
    return f"""
        question: {question}\n
        answer : {answer}\n
        ground_truth : {ground_truth}
    """


def test_model_on_one_question(
    raw_model,
    ft_model,
    test_conv: dict,
) -> str:
    """
    Asks the model for a question in the dataset and returns the result in a str.
    :param raw_model: the source model, not fine tuned in a hugging face pipeline
    :param ft_model: the fine tuned model in a hugging face pipeline
    :param test_conv: a chat between a user and an assistant, in classical open ai's chat template
    :return: a str containing the question, the expected answer, 2 samples of the answer
     by the fine tuned model and 1 sample of the answer by the raw model (not fine tuned).
    """
    input_chat = [test_conv[0]]
    question = test_conv[0]["content"]

    answer_no_fine_tuning = raw_model(input_chat)[0]["generated_text"][1]["content"]
    answer_fine_tuning = ft_model(input_chat)[0]["generated_text"][1]["content"]
    answer_fine_tuning_2 = ft_model(input_chat)[0]["generated_text"][1]["content"]
    ground_truth = test_conv[1]["content"]
    return f"""
        question: {question}\n
        answer_no_fine_tuning : {answer_no_fine_tuning}\n
        answer_fine_tuning : {answer_fine_tuning}\n
        answer_fine_tuning_2 : {answer_fine_tuning_2}\n
        ground_truth : {ground_truth}
    """


class CustomCallback(TrainerCallback):
    """
    Callback that asks the model for an answer during the training
    """

    def __init__(self, raw_model_pipeline, ft_model_pipeline, test_conv) -> None:
        super().__init__()
        self.raw_model_pipeline = raw_model_pipeline
        self.ft_model_pipeline = ft_model_pipeline
        self.test_conv = test_conv

    def on_epoch_end(self, args, state, control, **kwargs):
        print(
            test_model_on_one_question(
                raw_model=self.raw_model_pipeline,
                ft_model=self.ft_model_pipeline,
                test_conv=self.test_conv,
            )
        )


class CustomCallbackSimple(TrainerCallback):
    """
    Callback that asks the model for an answer during the training
    """

    def __init__(self, pipeline, test_conv) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.test_conv = test_conv

    def on_epoch_end(self, args, state, control, **kwargs):
        print(
            test_one_model_on_one_question(
                model=self.pipeline,
                test_conv=self.test_conv,
            )
        )


import requests
import json
import asyncio
import aiohttp
import time


class WebUIConnector:
    """
    Simple connector that uses the python requests lib and the API of Open Web UI to get
    an easy access to a remote LLM; and provides formatted answer (python objects).
    """

    def __init__(
        self,
        token: str | None,
        url: str,
        fav_model: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
    ):
        if token is None:
            raise ValueError("Token for Open Web UI is None. Please specify a token.")
        self.token = token
        self.url = url
        self.fav_model = fav_model

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def get_json_chat_response(
        self, prompt: str, return_list: bool = False, model: str | None = None
    ) -> list | dict | str | None:
        """
        Makes an authenticated request to OWUI API as a chat. Parse the result into a json,
        and returns the json in a python object.
        """
        result = self.get_chat_response(prompt, model)
        if result is None:
            return
        if return_list:
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                print(
                    "Error during parsing result of request to json. Trying to remove ```json ```."
                )
            try:
                result = remove_json_markers(result)
                result = json.loads(result)
                print("Successfully parsed json.")
                return result
            except json.JSONDecodeError:
                print("Failed to parse result of request to json, skipping this.")
            return []
        return result

    def get_chat_response(self, prompt: str, model: str | None = None) -> str | None:
        """
        Acts as a chat : makes a authenticated request to OWUI API, and returns the
        answer in a str.
        """
        if model is None:
            model = self.fav_model
        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response: requests.Response = requests.post(
            self.url, json=body, headers=self.headers
        )

        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
        else:
            print("Error:", response.status_code, response.text)
            return
        return result

    async def fetch(self, session, body):
        async with session.post(self.url, json=body, headers=self.headers) as response:
            return await response.json()  # or response.text() for plain text

    async def fetch_all(self, convs):
        async with aiohttp.ClientSession() as session:
            results = []
            for each_conv in convs:
                body = {"model": self.fav_model, "messages": each_conv}
                results.append(self.fetch(session, body))
            return await asyncio.gather(*results)

    def fetch_multiple_conv_results(self, convs):
        st = time.time()
        all_res = asyncio.run(self.fetch_all(convs))
        et = time.time()
        total_time = et - st
        n_requests = len(convs)
        if (
            n_requests / total_time > 50
        ):  # avoid kick of the API in next calls (too late for those calls though ;) )
            time.sleep(10)
        results = []
        for each_res in all_res:
            try:
                response = each_res["choices"][0]["message"]["content"]
            except Exception as e:
                response = ""
                print(f"Error raised during extraction of answer : {each_res}. {e}")
            results.append(response)
        return results

    def conv_completion(self, convs):
        body: dict = {
            "model": self.fav_model,
            "messages": convs,
        }
        response: requests.Response = requests.post(
            self.url, json=body, headers=self.headers
        )
        return response


def remove_json_markers(input_string):
    if input_string.startswith("```json") and input_string.endswith("```"):
        return input_string[len("```json") : -len("```")]
    return input_string
