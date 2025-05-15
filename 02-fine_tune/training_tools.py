from transformers.pipelines.base import Pipeline
from transformers.trainer_callback import TrainerCallback


def test_model_on_one_question(
    raw_model: Pipeline,
    ft_model: Pipeline,
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


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


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
