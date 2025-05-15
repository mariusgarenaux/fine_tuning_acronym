def create_judgement_prompt(question, answer_to_test, definition, verbose_def=None):
    """
    Custom prompt to use a chatbot as a judge.
    """
    if verbose_def is None:
        additional_info = ""
    else:
        additional_info = (
            f"More verbose definition to assess the answer : {verbose_def}"
        )

    return f"""
    You are an evaluator, whose aim is to determine whether a given answer contains appropriate information about a given question. To know if the answer accurately addresses the question, you will be given a definition that must be contained into the answer to validate its accuracy. State your result in between the following tags : <result></result>. The result must be either 0 or 1. 0 stands for an inaccurate answer, and 1 for an accurate answer.

    Here is an example : 
    Question : “What is Juropa ?“
    Answer to test : “Juropa is the fourth moon of Jupiter”,
    Definition to assess the answer : “Julich Research on Petaflop Architectures”.
    Here, the result would be that the answer is’nt accurate. A you would have answered : <result>0</result>.

    Here is an other example, successful this time : 
    Question : “What is Juropa ?“
    Answer to test : “Juropa stands for Julich Research on Petaflop Architectures”,
    Definition to assess the answer : “Julich Research on Petaflop Architectures”.
    Here, the result would be that the answer is accurate. A you would have answered : <result>1</result>.

    Now, it’s your turn : 
    Question : “What does the acronym BPMN represent?”
    Answer to test : “BPMN stands for Business Process Model and Notation. It is a standardized language for modeling business”
    Definition to assess the answer : “Business Process Model and Notation”.
    {additional_info}
    Please state your result between two tags : <result> and </result>. 1 is for an accurate answer, 0 for a inaccurate one.
    """
