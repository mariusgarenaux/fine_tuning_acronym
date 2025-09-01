from metaflow import Runner

lrs = [3e-05, 7e-05, 1e-04, 5e-04, 1e-03]
epochs = [1, 2, 3, 4, 5]

for epoch in epochs:
    for lr in lrs:
        with Runner("main.py").run(learning_rate=lr, n_epochs=epoch) as running:
            print(f"Running {running.run}")
