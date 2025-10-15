# ViTsGraphicalPerception

![Graphical Abstract](src/Images/GP.png)

**Evaluating graphical perception capabilities of Vision Transformers**  
Vision Transformers (ViTs) have emerged as a powerful alternative to convolutional
neural networks (CNNs) in a variety of image-based tasks. While CNNs have pre-
viously been evaluated for their ability to perform graphical perception tasks, which
are essential for interpreting visualizations, the perceptual capabilities of ViTs remain
largely unexplored. In this work, we investigate the performance of ViTs in elementary
visual judgment tasks inspired by Cleveland and McGill’s foundational studies, which
quantified the accuracy of human perception across different visual encodings. Inspired
by their study, we benchmark ViTs against CNNs and human participants in a series of
controlled graphical perception tasks. Our results reveal that, although ViTs demonstrate
strong performance in general vision tasks, their alignment with human-like graphical
perception in the visualization domain is limited. This study highlights key perceptual
gaps and points to important considerations for the application of ViTs in visualization
systems and graphical perceptual modeling.

##  Repository structure
The src directory contains the code necessary to produce the data, train the models for all results reported in the paper.

### Installation 
#### Clone and set up a virtual environment
```bash
git clone git@github.com:poonam2308/ViTsGraphicalPerception.git
cd ViTsGraphicalPerception
bash setup_venv.sh
source venv/bin/activate

```
#### Manual set up alternative
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

#### How it works 
- Stimulus Generation (Data): [src/ClevelandMcGill](src/ClevelandMcGill) modules to build task specific images 
- Network: [src/Models](src/Models) modules to define the three types (CvT, Swin, vViT) network architecture used in the paper
- Training: [src/Experiments](src/Experiments) modules to perform training on CvT, Swin and vVit on generated data. Please data is generated during the training process and it is not saved in the disk. It can be easily produced with the stimuli generation step. 
- Evaluation: [src/TestEvaluation](src/TestEvaluation) modules to evaluate the trained checkpoints (weights) on the test dataset. 
- Analysis: [src/Analysis](src/Analysis) modules to compare CvT, Swin, and vViT to human performance on the same synthetic stimuli.

#### Weights
The trained checkpoints for the experiments are available on GoogleDrive
Access it [here](https://drive.google.com/drive/folders/16w2oXF3nrA5wI-i6CxIxIX73Z7Pf6qWF?usp=drive_link)


### Run Experiments

1. **Create required folders**

```bash
   mkdir -p src/Experiments/chkpt
   mkdir -p src/Experiments/trainingplots
```

2. **Train a model (per task/script)**

```bash
   python src/Experiments/<name_of_file>.py
```
3. **Customize data sizes, batch size, epochs**
All experiment scripts accept the following flags:

--train_target <int> number of training samples

--val_target <int> number of validation samples

--test_target <int> number of test samples

--batch_size <int> batch size

--epochs <int> number of training epochs

```bash
  python src/Experiments/cvt_bfr.py --train_target 100 --val_target 20 --test_target 20 --batch_size 32  --epochs 100
```


###  Evaluation and Analysis
Reproduce the following via single scripts

**Analysis figures Run the analysis scripts directly**

1. Naviagte to the direcory [src/Analysis](src/Analysis)
2. Open and run:

```bash
Analysis.ipynb
```

**Baseline evaluation (all models via one single script notebook)**
1. Naviagte to the directory [src/TestEvaluation](src/TestEvaluation) 
  Download the pretrained checkpoints and place them in:

```bash
TestEvaluation/chkpt/
```
2. Open and run:

```bash
Main_Evaluation.ipynb
```

All evaluation results are saved as CSV files under the  results/ subfolders.


**Ablation study (all models via one single script notebook)**

1. Open and run:

```bash
Ablation_Evaluation.ipynb
```