This part is about the evaluation of the trained model. 

First, it is advised to run it with the raw model (that is, without fine-tuning) to have ground result. 

Then you can try with your finetuned models to compare them and watch how much they have learned of the training dataset. 


## Requirements 

To run this section, you'll need :

- a model (!), and its location. Once again, it will depend on your infrastructure; but they are always located in a folder named 'model' inside a bucket.

- an evaluation dataset, that has the same structure as the training one. Normally, it was created alongside the training dataset and is located in the same place. If you don't have one, you can either re-run the [01-create_dataset](../01-create_dataset) part; or copy-paste the given one which is located here : [../example_data/eval_dataset.json](../example_data/eval_dataset.json). The paste location depends on your computing infrastructure : 

    - on Datalab GCP : here : [../../bucket/fine_tuning_acronym/data/eval_dataset.json](../../bucket/fine_tuning_acronym/data/eval_dataset.json),

    - on Onyxia or locally; here [../bucket/data/eval_dataset.json](../bucket/data/eval_dataset.json)

- an access to Open Web UI (as in section 01-create_dataset), to use a LLM as a judge of the one you just fine-tuned. See [../README.md](../README.md) in Getting Started part according to your computing infra.

## Structure

You'll find here 3 notebooks : 

- [0-inference.ipynb](0-inference.ipynb) : small one, just to load a model and use it for inference to try it (because human test is important !)

- [1-test.ipynb](1-test.ipynb) : big one, here we load the model and try it on the evaluation dataset. The results are stored in the bucket, in the `tests` folder. 

- [2-compare_models](2-compare_models.ipynb) : here, no need to load the models, we just load the test result of several fine-tuned models and compare them to see the impact of tweaking training parameters on the test results.