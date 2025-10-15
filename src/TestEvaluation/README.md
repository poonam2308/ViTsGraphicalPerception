# Evaluation 

- Download the model checkpoints and place them in the specified folders ('src/Test/Evaluation/chkpt') 
- Launch the app (Gradio UI)
  - This starts a Gradio interface where you can select a model and a task.
  - After you click Run, the app evaluates the selected configuration and reports the MLAE metric (the paper’s evaluation metric).
  - The app writes results to a 'results/' folder.
- Each additional script in the repository corresponds to an evaluation module for a specific ablation study.
- Statisical test can be verified from notebook [Statistical_Test](Statistical_Test.ipynb)
- Checkpoints: total size is ~96 GB. Please email the author to request the specific ablation checkpoints you need and they will be shared individually