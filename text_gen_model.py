import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from typing import Tuple, Dict, Any, override


class AcronymTokenizer:
    def __init__(self, model_name, torch_dtype):
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.model = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.pad_token = self.model.eos_token

    def encode_acronym(self, all_acro: list[dict]):
        all_convs = [
            [
                {
                    "role": "user",
                    "content": f"What does {acro_dict['acronym']} means ?",
                },
                {
                    "role": "assistant",
                    "content": f"{acro_dict['acronym']} stands for {acro_dict["definition"]}",
                },
            ]
            for acro_dict in all_acro
        ]
        encoding = self.model.apply_chat_template(
            conversation=all_convs,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            padding=True,
            max_length=256,
        )
        return encoding


class AcronymDataset(Dataset):
    def __init__(self, encodings: dict):
        self.encodings: dict = encodings

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

    def __len__(self):
        return len(self.encodings)


class InstructionTextGenerationPipeline:
    def __init__(
        self,
        model_name: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "mps",
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )

        self.acronym_tokenizer = AcronymTokenizer(model_name, torch_dtype)
        self.tokenizer = self.acronym_tokenizer.model

        # if tokenizer.pad_token_id is None:
        #     warnings.warn(
        #         "pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id."
        #     )
        #     tokenizer.pad_token = tokenizer.eos_token

        self.device = torch.device(device)
        self.model.eval()
        self.model.to(device=device, dtype=torch_dtype)
        self.torch_dtype = torch_dtype
        self.generate_kwargs = {
            "temperature": 0.5,
            "top_p": 0.92,
            "top_k": 0,
            "max_new_tokens": 512,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.1,  # 1.0 means no penalty, > 1.0 means penalty, 1.2 from CTRL paper
        }

    def __call__(
        self, text_input, **generate_kwargs: Dict[str, Any]
    ) -> Tuple[str, str, float]:

        input_ids = self.tokenizer.encode(text_input, return_tensors="pt").to(
            self.device
        )

        gkw = {**self.generate_kwargs, **generate_kwargs}
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gkw)
        # Slice the output_ids tensor to get only new tokens
        new_tokens = output_ids[0, len(input_ids[0]) :]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return output_text
