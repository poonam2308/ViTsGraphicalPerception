import argparse
import wandb

# usage of arg par
def get_args_parser():
    parser = argparse.ArgumentParser("Arguments for pretrained models")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size (default : 64)")

    parser.add_argument("--NOISE",
                        action="store_true",
                        default=True,
                        help="Add noise to the data")

    parser.add_argument("--epochs", default=100, type=int, help="training epochs")

    # Optimizer parameter
    parser.add_argument("--weight_decay",
                        type=float,
                        default=1e-6,
                        help="weight decay (default: 0.000001)")

    parser.add_argument("--momentum",
                        type=float,
                        default=0.9,
                        help="momentum (default: 0.9)")

    parser.add_argument("--nesterov",
                        action="store_true",
                        default=True,
                        help="used in SGD (default : False)")

    parser.add_argument("--lr",
                        type=float,
                        default=0.0001,
                        metavar="LR",
                        help="learning rate (lr)")

    #data gen values
    parser.add_argument("--train_target", default=60000, type=int, help="training data")
    parser.add_argument("--val_target", default=20000, type=int, help="validation data")
    parser.add_argument("--test_target", default=20000, type=int, help="testing data")

    #vit model parameters
    parser.add_argument("--vit_input_size", default=224, type=int, help="images input size")
    parser.add_argument("--vit_patch_size", default=16, type=int, help="patch size")
    parser.add_argument("--vit_dim", default=512, type=int, help="hidden dim size for vit")
    parser.add_argument("--vit_depth", default=8, type=int, help="depth for vit")
    parser.add_argument("--vit_heads", default=4, type=int, help="MHA for vit")
    parser.add_argument("--vit_mlp_dim", default=1024, type=int, help="MLP dim for vit")
    parser.add_argument("--num_classes", default=1, type=int, help="number of classes for vit, cvt")

    #swin_model parameters
    parser.add_argument("--swin_hidden_dim", default=96, type=int, help="hidden dim for swin")
    parser.add_argument("--swin_channels", default=1, type=int, help="image channels for swin")
    parser.add_argument("--swin_head_dim", default=32, type=int, help="head dim for swin")
    parser.add_argument("--window_size", default=7, type=int, help="window size for swin")
    parser.add_argument("--relative_pos_embedding", action= "store_true", default=False,help="relative pos for swin")
    parser.add_argument("--num_outputs", default=1, type=int, help="number of classes swin")

    parser.add_argument("--num_channels", default=3, type=int, help="image channels for transformers")

    #wandb arguments
    parser.add_argument("--wandb_tags", nargs="+", default=[])
    parser.add_argument("--group", type=str, default="graphical_perception", help="wandb group keyword")
    parser.add_argument("--log_code", action="store_true", help="log code as wandb artifact")

    return parser

# usage of weight and bias
def init_wandb(config=None, tags=None, group="graphical_perception"):
    wandb.init(
        project="GraphicalPerception",
        config=config,
        tags=tags,
    )
    wandb.define_metric("train/accuracy", summary="max")
    wandb.define_metric("val/acc1", summary="max")

