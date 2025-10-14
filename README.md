# ViTsGraphicalPerception

![Graphical Abstract](src/Images/graphical_abstract.pdf)

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
```commandline
git clone git@github.com:poonam2308/ViTsGraphicalPerception.git
cd ViTsGraphicalPerception
bash setup_venv.sh
source .venv/bin/activate
pip install -r requriements.txt

```
#### Manual set up alternative
```commandline
python -m venv .venv
source .venv/bin/activate
pip install -r requirments.txt

```

#### How it works 
- Stimulus Generation (Data): 'src/ClevelandMcGill' modules to build task specific images 
- Network: 'src/Models' modules to define the three types (CvT, Swin, vViT) network architecture used in the paper
- Training: 'src/Experiments' modules to perform training on CvT, Swin and vVit on generated data. 
- Evaluation: 'src/TestEvaluation' modules to evaluate the trained checkpoints (weights) on the test dataset. 
- Analysis: 'src/Analysis' modules to compare CvT, Swin, and vViT to human performance on the same synthetic stimuli.

#### Weights
The trained checkpoints for the experiments are available on GoogleDrive
Access it [here](link)