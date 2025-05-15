def create_acronym_prompt(n_conv, acro, definition, verbose_def=None):
    """
    Custom prompt to get a formatted result synthethic conversation about acronym
    (or simply an unknown term) and definitions.
    """

    additional_info = (
        f" Here is some additional information about this acronym {verbose_def}."
        if verbose_def is not None
        else ""
    )
    return (
        f"Create {n_conv} fictive conversations between an user and an assistant.\n"
        "Those conversations must contains 1 question and 1 answer.\n"
        f"Each question must be an user asking for the definition the term {acro}; and each answer must contain the definition : '{definition}'.{additional_info}\n"
        "All the conversations must be somehow diverse.\n"
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
        "Keep it short.\n"
        "Your final answer must be only the raw json; no fioritures.\n"
    )
