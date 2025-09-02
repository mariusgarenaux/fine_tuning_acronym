
## Training dataset

The previously generated data must be in your bucket, whose location change according to the computing infrastructure :

- on Datalab GCP : you'll find the training dataset here : [../../bucket/fine_tuning_acronym/data/train_dataset.json](../../bucket/fine_tuning_acronym/data/train_dataset.json),

- on Onyxia or locally; you'll find the training dataset here : : [../bucket/data/train_dataset.json](../bucket/data/train_dataset.json).

If you don't have one, you can simply copy-paste to the above location the example one : [../example_data/train_dataset.json](../example_data/train_dataset.json); or create one by going to [01-create_dataset](../01-create_dataset/).

To start the training, go to the notebook [training.ipynb](training.ipynb) !