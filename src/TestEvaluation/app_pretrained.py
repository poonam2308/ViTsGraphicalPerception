import json
import os
import gradio as gr
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("../../")
from torchvision import transforms
from src.Models.cvt import CvTRegression
from src.Models.one_epoch_run import testingEpoch, testingEpochOne
from src.Models.swin import SwinRegression
from src.Models.vit import ViTRegression
import numpy as np
from src.ClevelandMcGill.figure1 import Figure1
from src.ClevelandMcGill.figure12 import Figure12
from src.ClevelandMcGill.figure3 import Figure3
from src.ClevelandMcGill.figure4 import Figure4
from src.ClevelandMcGill.weber import Weber

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.Datasets.testdataset import TestDataset, test_normalization_data, \
    test_pl_data_generation, test_pa_data_generation, test_bfr_data_generation, test_wb_data_generation, \
    test_reg_data_generation

TASKS = ['Elementary Perceptual Task',
         'Position-Length',
         'Position-Angle',
         'Bar and Framed Rectangle',
         'Webers Law']

EL_DATATYPE_LIST = ['position_common_scale',
                    'position_non_aligned_scale',
                    'length',
                    'direction',
                    'angle',
                    'area',
                    'volume',
                    'curvature',
                    'shading']
# position length
PL_DATATYPE_LIST = ['data_to_type1',
                    'data_to_type2',
                    'data_to_type3',
                    'data_to_type4',
                    'data_to_type5']

# position angle
PA_DATATYPE_LIST = ['data_to_barchart',
                    'data_to_piechart',
                    'data_to_piechart_aa']

# bar frame rectanlges
BFR_DATATYPE_LIST = ['data_to_bars',
                     'data_to_framed_rectangles']

# weber law
WB_DATATYPE_LIST = ['base10',
                    'base100',
                    'base1000']

DATATYPE_LIST = {TASKS[0]: EL_DATATYPE_LIST,
                 TASKS[1]: PL_DATATYPE_LIST,
                 TASKS[2]: PA_DATATYPE_LIST,
                 TASKS[3]: BFR_DATATYPE_LIST,
                 TASKS[4]: WB_DATATYPE_LIST}


def data_gen(task_name, transform, experiment_type):
    if task_name == "Elementary Perceptual Task":
        DATATYPE = eval('Figure1.' + experiment_type)
        X_test, y_test = test_reg_data_generation(DATATYPE, NOISE=True, test_target=20000)
        X_test = test_normalization_data(X_test)
        y_test = test_normalization_data(y_test)
        X_test -= 0.5
        test_dataset = TestDataset(X_test, y_test, transform=transform, channels=True)
        return test_dataset
    elif task_name == "Position-Length":
        DATATYPE = eval('Figure4.' + experiment_type)
        X_test, y_test = test_pl_data_generation(DATATYPE, NOISE=True, test_target=20000)
        X_test = test_normalization_data(X_test)
        y_test = test_normalization_data(y_test)
        X_test -= 0.5
        test_dataset = TestDataset(X_test, y_test, transform=transform, channels=True)
        return test_dataset
    elif task_name == "Position-Angle":
        DATATYPE = eval('Figure3.' + experiment_type)
        X_test, y_test = test_pa_data_generation(DATATYPE, NOISE=True, test_target=20000)
        X_test = test_normalization_data(X_test)
        y_test = test_normalization_data(y_test)
        X_test -= 0.5
        test_dataset = TestDataset(X_test, y_test, transform=transform, channels=True)
        return test_dataset
    elif task_name == "Bar and Framed Rectangle":
        DATATYPE = eval('Figure12.' + experiment_type)
        X_test, y_test = test_bfr_data_generation(DATATYPE, NOISE=True, test_target=20000)
        X_test = test_normalization_data(X_test)
        y_test = test_normalization_data(y_test)
        X_test -= 0.5
        test_dataset = TestDataset(X_test, y_test, transform=transform, channels=True)
        return test_dataset
    elif task_name == "Webers Law":
        DATATYPE = eval('Weber.' + experiment_type)
        X_test, y_test = test_wb_data_generation(DATATYPE, NOISE=True, test_target=20000)
        X_test = test_normalization_data(X_test)
        y_test = test_normalization_data(y_test)
        X_test -= 0.5
        test_dataset = TestDataset(X_test, y_test, transform=transform, channels=True)
        return test_dataset
    else:
        return 'please select a task.'


def data_loader(task_name, experiment_type):
    transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])
    batch_size = 64
    test_dataset = data_gen(task_name, transform, experiment_type)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    return test_loader


def test_model(model, model_name, task_name, exp_type, test_loader):
    model.to(device)
    PATH = "chkpt/chkpts_fromCluster/channels3/" + model_name.lower() + "3channels_" + exp_type + ".pth"
    model.load_state_dict(torch.load(PATH))
    if task_name == "Position-Length" or task_name == "Position-Angle" or "Bar and Framed Rectangle":
        m_error = testingEpoch(model, test_loader, device)
    else:
        m_error = testingEpochOne(model, test_loader, device)
    json_data = {'Model': model_name, 'Task_name': task_name, 'Experiment_type': exp_type, 'MLAE': m_error}
    # Write JSON file
    file_name = 'results/' + model_name + 'pretrained' + exp_type + '.json'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', ) as outfile:
        json.dump(json_data, outfile)
    return m_error


def testing_model(model_name, task_name, exp_type):
    TWO_OUTPUTS = 2
    FIVE_OUTPUTS = 5
    SINGLE_OUTPUT = 1
    test_loader = data_loader(task_name, exp_type)
    if task_name == "Elementary Perceptual Task":
        if model_name == "CVT":
            model = CvTRegression(num_classes=SINGLE_OUTPUT, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
        elif model_name == "VIT":
            model = ViTRegression(num_classes=SINGLE_OUTPUT, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
        else:
            model = SwinRegression(num_outputs=SINGLE_OUTPUT, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
    elif task_name == "Position-Length":
        if model_name == "CVT":
            model = CvTRegression(num_classes=FIVE_OUTPUTS, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
        elif model_name == "VIT":
            model = ViTRegression(num_classes=FIVE_OUTPUTS, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
        else:
            model = SwinRegression(num_outputs=FIVE_OUTPUTS, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
    elif task_name == "Position-Angle":
        if model_name == "CVT":
            model = CvTRegression(num_classes=FIVE_OUTPUTS, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
        elif model_name == "VIT":
            model = ViTRegression(num_classes=FIVE_OUTPUTS, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
        else:
            model = SwinRegression(num_outputs=FIVE_OUTPUTS, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
    elif task_name == "Bar and Framed Rectangle":
        if model_name == "CVT":
            model = CvTRegression(num_classes=TWO_OUTPUTS, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
        elif model_name == "VIT":
            model = ViTRegression(num_classes=TWO_OUTPUTS, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
        else:
            model = SwinRegression(num_outputs=TWO_OUTPUTS, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
    else:
        if model_name == "CVT":
            model = CvTRegression(num_classes=SINGLE_OUTPUT, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
        elif model_name == "VIT":
            model = ViTRegression(num_classes=SINGLE_OUTPUT, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE
        else:
            model = SwinRegression(num_outputs=SINGLE_OUTPUT, channels=3)
            MLAE = test_model(model, model_name, task_name, exp_type, test_loader)
            return MLAE


MODELS = ["CVT", "SWIN", "VIT"]


def rs_change(rs):
    return gr.update(choices=DATATYPE_LIST[rs], value=None)


with gr.Blocks() as app:
    model_name = gr.Dropdown(label="Model", choices=MODELS, value=MODELS[0])

    rs = gr.Dropdown(label="Task", choices=TASKS, value=TASKS[0])
    rs_hw = gr.Dropdown(label="Experiment Type", choices=DATATYPE_LIST[TASKS[0]], interactive=True)
    rs.change(fn=rs_change, inputs=[rs], outputs=[rs_hw])

    MLAE = gr.Text(label="MLAE score")
    btn = gr.Button("Evaluate")
    btn.click(testing_model, inputs=[model_name, rs, rs_hw], outputs=[MLAE])

app.launch()
