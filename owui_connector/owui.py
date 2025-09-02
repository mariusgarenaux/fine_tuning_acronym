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


def extract_json_from_answer(answer):
    """
    Structure the output (string) of ollama into
    a python list or dictionnary (json).
    """
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        print(
            "Error during parsing result of request to json. Trying to remove ```json ```."
        )
    try:
        answer = remove_json_markers(answer)
        answer = json.loads(answer)
        print("Successfully parsed json.")
        return answer
    except json.JSONDecodeError:
        print("Failed to parse result of request to json, skipping this.")
    return None
