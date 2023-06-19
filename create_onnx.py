#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import datetime
import os
import subprocess

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
        '--dataset', '-d',
        type=str,
        required=False,
        default='/media/ubuntu/G/dataset/data_odometry_velodyne/dataset',
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        required=False,
        default="1",
        help='Directory to put the predictions. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=False,
        default="/media/ubuntu/G/projects/net-3d/pointcloud-seg/CENet/CENet-main-base/models_4class_nopre",
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default='valid',
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("infering", FLAGS.split)
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file from %s" % FLAGS.model)
        ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
        os.makedirs(os.path.join(FLAGS.log, "sequences"))
        for seq in DATA["split"]["train"]:
            seq = '{0:02d}'.format(int(seq))
            print("train", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
        for seq in DATA["split"]["valid"]:
            seq = '{0:02d}'.format(int(seq))
            print("valid", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
        for seq in DATA["split"]["test"]:
            seq = '{0:02d}'.format(int(seq))
            print("test", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise

    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    # create user and infer dataset
    user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model,FLAGS.split)
    # user.infer()

    model = user.model
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
    onnx_path = os.path.join(FLAGS.model, "model_ce_train_test.onnx")
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

    # onnx_path = os.path.join(FLAGS.model, "model_modify.onnx")
    # onnx.save(model_onnx, onnx_path)
    # onnx.checker.check_model(model_onnx)

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model_onnx.graph))
