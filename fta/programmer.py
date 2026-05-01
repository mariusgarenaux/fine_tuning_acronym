# An example runner that runs the workflow with
# various learning rates

from metaflow.runner.metaflow_runner import Runner

lrs = [3e-05, 7e-05, 1e-04, 5e-04, 1e-03]
n_epochs = 5

for lr in lrs:
    with Runner("main.py").run(learning_rate=lr, n_epochs=n_epochs) as running:
        print(f"Running {running.run}")
