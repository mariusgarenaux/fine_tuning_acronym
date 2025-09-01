import asyncio
import aiohttp
import requests

from owui_connector.owui import WebUIConnector
from tqdm import tqdm

owui_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjQ2YWIzMmNkLWIyNjItNGY2Mi1iYWQ4LWFkYzEzMTg5OGQ0MyJ9.3uYG1HMdRg9i9uGLx8OvqMK8udqmi-1IFNcZLTqbHuw"
owui_url = "https://ragarenn.eskemm-numerique.fr/marius.garenaux-gruau@irisa.fr/api/chat/completions"
owui_fav_model = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"

owui = WebUIConnector(owui_token, owui_url, fav_model=owui_fav_model)

convs = [
    [{"role": "user", "content": "Why is the sky blue ?"}],
    [{"role": "user", "content": "Why is the sky red ?"}],
    [{"role": "user", "content": "Why is the sky yellow ?"}],
    [{"role": "user", "content": "Why is the sky blue ?"}],
]


if __name__ == "__main__":
    response = owui.fetch_multiple_conv_results(convs)
    print(response)
