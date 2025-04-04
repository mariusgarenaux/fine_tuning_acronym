import requests
import json


class WebUIConnector:
    """
    Simple connector that uses the python requests lib and the API of Open Web UI to get
    an easy access to a remote LLM.
    """

    def __init__(
        self,
        token: str,
        url: str,
        fav_model: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
    ):
        self.token = token
        self.url = url
        self.fav_model = fav_model

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def get_chat_response(
        self, prompt: str, return_list: bool = False, model: str | None = None
    ) -> list | None:
        """
        Acts as a chat : makes a authenticated request to OWUI API; and returns the answer as str
        """
        if model is None:
            model = self.fav_model
        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(self.url, json=body, headers=self.headers)

        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
        else:
            print("Error:", response.status_code, response.text)
            return
        if return_list:
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                print(
                    "Error during parsing result of request to json, skipping result."
                )
            return []
        return result


def create_acronym_prompt(n_conv, acro, definition, definition_verbose):
    if definition is None:
        definition = definition_verbose

    if definition_verbose is None:
        definition_verbose = definition
    return (
        f"Create {n_conv} fictive conversations between an user and an assistant.\n"
        "Those conversations must contains 1 question and 1 answer.\n"
        f"Each question must be an user asking for the definition of the acronym {acro}; and each answer must contain the definition : '{definition}'; or a more verbose definition : {definition_verbose}.\n"
        "All the answer must be somehow diverse.\n"
        "Each conversation will be formatted in a json list, where each element is itself a list of the form : \n"
        "[\n"
        "  {\n"
        "     'role': 'user'',\n"
        "     'content': THE QUESTION\n"
        "  },\n"
        "  {\n"
        "    'role': 'assistant',\n"
        "     'content': THE ANSWER\n"
        "  }\n"
        "] \n"
        "Keep it short. The answer must be the raw json; no fioritures.\n"
    )
