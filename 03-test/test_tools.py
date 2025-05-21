import re


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
