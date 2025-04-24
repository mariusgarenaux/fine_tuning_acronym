# Fine Tuning LLM

Objectif : fine-tuning de Llama pour apprendre les acronymes et leurs définitions


# To run on datalab-gcp :

run the script init_gcp.sh :

```bash
bash init_gcp.sh
```
    
-> install missing python libs with the python used in jupyter

-> asks for hugging face token to retrieve restricted models

set datalab boolean to True in notebook; and checkpoint_path str

create a data folder with boosted_data.json

run cells
