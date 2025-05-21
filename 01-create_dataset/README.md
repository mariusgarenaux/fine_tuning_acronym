This part aims to create a training dataset of fake conversations about a given list of acronyms and their definitions. 

The idea is to start from a list of acronyms and their definitions, for example :

```json
{
    "acronym": "TOAST",
    "definition": "Techniques for Outstanding Appetizing Sauces and Treats"
}
```

and to ask an instruct LLM to create fake conversations with it.

The list of acronyms used in this notebook can be accessed here : [acronym.json](../example_data/acronym.json). However, if you want to change it to add your own acronyms, you'll need to modify a copy of it, which location depends on your computing infrastructure :

- [Datalab GCP](../../bucket/fine_tuning_acronym/data/acronym.json),

- [Onyxia](../bucket/data/acronym.json),

- [Locally](../bucket/data/acronym.json).