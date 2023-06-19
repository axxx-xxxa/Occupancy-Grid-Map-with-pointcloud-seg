#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch.nn as nn
import torch
import argparse
import yaml
import os
import torch.nn.utils.prune as prune

def get_model():
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")

    parser.add_argument(
        '--root', '-r',
        type=str,
        required=False,
        default="/media/ubuntu/G/projects/net-3d/pointcloud-seg/CENet/CENet-main-base/logs/part1test_pre_homedecay/test/",
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=False,
        default="SENet",
        help='model.'
    )
    parser.add_argument(
        '--classes', '-s',
        type=int,
        required=False,
        default=5,
        help='classes'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("model", FLAGS.root + FLAGS.model)
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file from %s" % FLAGS.root)
        ARCH = yaml.safe_load(open(FLAGS.root + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.root)
        DATA = yaml.safe_load(open(FLAGS.root + "/data_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # does model folder exist?
    if os.path.isdir(FLAGS.root):
        print("model folder exists! Using model from %s" % (FLAGS.root))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    with torch.no_grad():
        torch.nn.Module.dump_patches = True
        if ARCH["train"]["pipeline"] == "hardnet":
            from modules.network.HarDNet import HarDNet

            model = HarDNet(FLAGS.classes + 1, ARCH["train"]["aux_loss"])

        if ARCH["train"]["pipeline"] == "res":
            from modules.network.ResNet import ResNet_34

            model = ResNet_34(FLAGS.classes + 1, ARCH["train"]["aux_loss"])

            def convert_relu_to_softplus(model, act):
                for child_name, child in model.named_children():
                    if isinstance(child, nn.LeakyReLU):
                        setattr(model, child_name, act)
                    else:
                        convert_relu_to_softplus(child, act)

            if ARCH["train"]["act"] == "Hardswish":
                convert_relu_to_softplus(model, nn.Hardswish())
            elif ARCH["train"]["act"] == "SiLU":
                convert_relu_to_softplus(model, nn.SiLU())

        if ARCH["train"]["pipeline"] == "fid":
            from modules.network.Fid import ResNet_34

            model = ResNet_34(FLAGS.classes + 1, ARCH["train"]["aux_loss"])

            if ARCH["train"]["act"] == "Hardswish":
                convert_relu_to_softplus(model, nn.Hardswish())
            elif ARCH["train"]["act"] == "SiLU":
                convert_relu_to_softplus(model, nn.SiLU())

        #     print(model)
        w_dict = torch.load(FLAGS.root + "/" + FLAGS.model,
                            map_location=lambda storage, loc: storage)
        model.load_state_dict(w_dict['state_dict'], strict=True)
        print(f"params: {sum(p.numel() for p in model.parameters())}")

    # # report model parameters
    # weights_total = sum(p.numel() for p in model.parameters())
    # weights_grad = sum(p.numel()
    #                    for p in model.parameters() if p.requires_grad)
    # print("Total number of parameters: ", weights_total)
    # print("Total number of parameters requires_grad: ", weights_grad)
    #
    # # convert to ONNX
    # dummy_input = torch.randn(1, 5,
    #                           64,
    #                           512, device='cuda')
    #
    # model = model.cuda().eval()
    # onnx_path = os.path.join(FLAGS.root, "model.onnx")
    # print("saving model in ", onnx_path)
    # with torch.no_grad():
    #     torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)

    return model

def prune_model_layer(model):
    module1 = model.conv1.conv
    weights_total = sum(p.numel() for p in module1.parameters())
    print("Total number of parameters: ", weights_total)

    # module2 = model.conv2.conv
    # module3 = model.conv3.conv
    prune.l1_unstructured(module1, name="weight", amount=0.3)
    # prune.l1_unstructured(module1, name="bias", amount=0.3)
    # prune.l1_unstructured(module2, name="weight", amount=0.3)
    # prune.l1_unstructured(module2, name="bias", amount=0.3)
    # prune.l1_unstructured(module3, name="weight", amount=0.3)
    # prune.l1_unstructured(module3, name="bias", amount=0.3)
    # print(list(module1.named_parameters()))
    # print(list(module1.named_buffers()))
    # model.conv1.conv = module1

    weights_total = sum(p.numel() for p in module1.parameters())
    print("Total number of parameters: ", weights_total)
def prune_model_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=0.2)
        elif isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name="weight", amount=0.4, n=2, dim=0)
    weights_total = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", weights_total)

if __name__ == '__main__':
    model = get_model()
    weights_total = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", weights_total)
    model1 = prune_model_layer(model)
    model2 = prune_model_modules(model)