#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import datetime
import os
import subprocess
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import onnx
import torch
import yaml
#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
from modules.user import User

if __name__ == '__main__':
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")

    parser.add_argument(
        '--root', '-r',
        type=str,
        required=False,
        default="/media/ubuntu/G/projects/net-3d/porintcloud-segmodels/pth_segmodels/CENet/CENet_Version_tingwei_homedecay/test/",
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=False,
        default="SENet_valid_best",
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
    
    
    
    
    
    # report model parameters
    weights_total = sum(p.numel() for p in model.parameters())
    weights_grad = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # convert to ONNX
    dummy_input = torch.randn(1, 5,
                              64,
                              512, device='cuda')
    # (Pdb) proj_in.shape
    # torch.Size([1, 5, 64, 2048])
    # (Pdb) proj_range.shape (also proj_range)
    # torch.Size([1, 64, 2048])

    model = model.cuda().eval()
    onnx_path = os.path.join(FLAGS.root, "model.onnx")
    print("saving model in ", onnx_path)
    with torch.no_grad():
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
        
    # check that it worked
    # model_onnx = onnx.load(onnx_path)
    # graph = model_onnx.graph
    # nodes = graph.node
    # graph.node.remove(nodes[1])
    # nodes[1].input[0] = '0'
    # graph.node.remove(nodes[0])

    # nodes[2].inputs = ['0']
    # new_node = onnx.helper.make_node(
    #     " ",
    #     inputs = ['0'],
    #     outputs = ['411']
    # )
    # graph.node.insert(1,new_node)
    # print("----------------node0: ",nodes[0])
    # print("----------------node1: ",nodes[1])
    # print("----------------node2: ",nodes[2])

    # onnx_path = os.path.join(FLAGS.root, "model_modify.onnx")
    # onnx.save(model_onnx, onnx_path)
    # onnx.checker.check_model(model_onnx)

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model_onnx.graph))
