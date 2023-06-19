#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import os
import numpy as np
from matplotlib import pyplot as plt

from postproc.KNN import KNN
from modules.ioueval import *
import cv2


def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)


def make_log_img(depth, mask, pred, gt, color_fn):
    # input should be [depth, pred, gt]
    # make range image (normalized to 0,1 for saving)
    depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX,
                           dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
    out_img = cv2.applyColorMap(
        depth, get_mpl_colormap('viridis')) * mask[..., None]
    # make label prediction
    pred_color = color_fn((pred * mask).astype(np.int32))
    out_img = np.concatenate([out_img, pred_color], axis=0)
    # make label gt
    gt_color = color_fn(gt)
    out_img = np.concatenate([out_img, gt_color], axis=0)
    out_img = out_img[-64:,:,:]
    return (out_img).astype(np.uint8)


def calculate_line_points_by_slope(a, b):
    x1, y1 = a
    x2, y2 = b

    points = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # 确定斜率
    if dx > dy:
        slope = dy / dx
        if x2 > x1:
            xi = 1
        else:
            xi = -1
        yi = slope * xi
    else:
        slope = dx / dy
        if y2 > y1:
            yi = 1
        else:
            yi = -1
        xi = slope * yi

    x, y = x1, y1
    points.append((x, y))

    # 根据斜率计算直线上的点
    while x != x2 or y != y2:
        if dx > dy:
            x += xi
            y += yi
        else:
            y += yi
            x += xi

        points.append((x, y))

    return points

def calculate_line_points_by_iter_r(a, b):

    lines_left = []
    lines_right = []

    init_step = 16
    coef_step = 0.20
    coef_ydis = 1 * (abs(a[1] - b[1]) / 11)
    coef_xdis = 1 * (abs(a[0] - b[0]) / 256)
    # print("---", coef_ydis)
    # print("---", coef_xdis)
    internal = 0
    # start 32 end 43
    internal_per_row = []
    y_dis = abs(a[1] - b[1])
    y_min = min(a[1], b[1])
    for i, cy in enumerate(range(y_min, y_min + y_dis)):
        internal += ((coef_ydis * coef_step * (cy)) + init_step) * coef_xdis
        # ego_coor_left = (int(max(a[0], b[0]) - internal), cy)
        # ego_coor_left = (cy, int(max(a[0], b[0]) - internal))
        ego_coor_right = (cy, int(max(a[0], b[0]) + internal))
        # print(f"({int(max(a[0], b[0]) - internal)},{cy})")
        lines_right.append(ego_coor_right)
        # lines_right.append(ego_coor_right)
    return lines_right

def calculate_line_points_by_iter_rb(a, b):

    lines_left = []
    lines_right = []

    init_step = 16
    coef_step = 0.20
    coef_ydis = 1 * (abs(a[1] - b[1]) / 11)
    coef_xdis = 1 * (abs(a[0] - b[0]) / 256)
    # print("---", coef_ydis)
    # print("---", coef_xdis)
    internal = 0
    # start 32 end 43
    internal_per_row = []
    y_dis = abs(a[1] - b[1])
    y_min = min(a[1], b[1])
    for i, cy in enumerate(range(y_min, y_min + y_dis)):
        internal += ((coef_ydis * coef_step * (cy)) + init_step) * coef_xdis
        # ego_coor_left = (int(max(a[0], b[0]) - internal), cy)
        # ego_coor_left = (cy, int(max(a[0], b[0]) - internal))
        ego_coor_right = (cy, int(max(a[0], b[0]) + internal))
        # print(f"({int(max(a[0], b[0]) - internal)},{cy})")
        lines_right.append(ego_coor_right)
        # lines_right.append(ego_coor_right)
    return lines_right

def calculate_line_points_by_iter_l(a, b):

    lines_left = []
    lines_right = []

    init_step = 16
    coef_step = 0.20
    coef_ydis = 1 * (abs(a[1] - b[1]) / 11)
    coef_xdis = 1 * (abs(a[0] - b[0]) / 256)
    # print("---", coef_ydis)
    # print("---", coef_xdis)
    internal = 0
    # start 32 end 43
    internal_per_row = []
    y_dis = abs(a[1] - b[1])
    y_min = min(a[1], b[1])
    for i, cy in enumerate(range(y_min, y_min + y_dis)):
        internal += ((coef_ydis * coef_step * (cy)) + init_step) * coef_xdis
        # ego_coor_left = (int(max(a[0], b[0]) - internal), cy)
        ego_coor_left = (cy, int(max(a[0], b[0]) - internal))
        # ego_coor_right = (cy, int(max(a[0], b[0]) + internal))
        # print(f"({int(max(a[0], b[0]) - internal)},{cy})")
        lines_left.append(ego_coor_left)
        # lines_right.append(ego_coor_right)
    return lines_left


def calculate_line_points_by_iter_lb(a, b):

    lines_left = []
    lines_right = []

    init_step = 16
    coef_step = 0.20
    coef_ydis = 1 * (abs(a[1] - b[1]) / 11)
    coef_xdis = 1 * (abs(a[0] - b[0]) / 256)
    # print("---", coef_ydis)
    # print("---", coef_xdis)
    internal = 0
    # start 32 end 43
    internal_per_row = []
    y_dis = abs(a[1] - b[1])
    y_min = min(a[1], b[1])
    for i, cy in enumerate(range(y_min, y_min + y_dis)):
        internal += ((coef_ydis * coef_step * (cy)) + init_step) * coef_xdis
        # ego_coor_left = (int(max(a[0], b[0]) - internal), cy)
        ego_coor_left = (cy, int(max(a[0], b[0]) - internal))
        # ego_coor_right = (cy, int(max(a[0], b[0]) + internal))
        # print(f"({int(max(a[0], b[0]) - internal)},{cy})")
        lines_left.append(ego_coor_left)
        # lines_right.append(ego_coor_right)
    return lines_left

def make_bev_img_front(bev_img ,gt_np):

    # 64 * 512  mask
    # 思路
    # 1. [h] 只关注车身周围10m的情况 那么mask中 大概[-32:,:]是有效的
    # 2. [w] 其中256为正前方  0/512为正后方  128为左  384为右
    # 3. 这意味着 在bev图中 我如果想映射正前方偏右一点点的一条直线 那么在projmask中应该取 256偏右的一条斜线
    # 4. bev先预设20*20（格栅） 用20条线从256开始 10往左 10往右 生成正前方bev 用另外20条线从0/512开始 10往0-> 10往512-< 生成正后方bev

    # 1. 生成20 * 2条线
    #  斜率变化要根据projmask来 想象成走路时路沿距离自己的角度变换
    front_up_left_x         = [x for x in range(256, 206, -5)]
    front_up_right_x        = [x for x in range(256, 306, 5)]
    front_down_left_x       = [x for x in range(256, 136, -12)]
    front_down_right_x      = [x for x in range(256, 376, 12)]
    front_up_y              = 32
    front_down_y            = 43
    front_up_left_coor      = [(x, front_up_y) for x in front_up_left_x]
    front_down_left_coor    = [(x, front_down_y) for x in front_down_left_x]
    front_up_right_coor     = [(x, front_up_y) for x in front_up_right_x]
    front_down_right_coor   = [(x, front_down_y) for x in front_down_right_x]
    front_points_left       = []
    front_points_right      = []
    epoch = 0
    for up_coor, down_coor in zip(front_up_left_coor, front_down_left_coor):
        # print(up_coor, down_coor)
        lines_left = calculate_line_points_by_iter_l(up_coor, down_coor)

        front_points_left.append(lines_left)
        epoch += 1

    for up_coor, down_coor in zip(front_up_right_coor, front_down_right_coor):
        # print(up_coor, down_coor)
        lines_right = calculate_line_points_by_iter_r(up_coor, down_coor)

        front_points_right.append(lines_right)
        epoch += 1

    for i, line in enumerate(front_points_left):
        for j, points in enumerate(line):
            print((9 - i, j), (points[0], points[1]), gt_np[points[0]][points[1]])
            if gt_np[points[0]][points[1]] == 4:
                bev_img[j][9-i] = 255
            elif gt_np[points[0]][points[1]] == 0:

                replace_gt = gt_np[points[0]+1][points[1]] or gt_np[points[0]+2][points[1]] or gt_np[points[0]+3][points[1]]
                print(f"replace_gt = {replace_gt} {gt_np[points[0]+1][points[1]]} {gt_np[points[0]+2][points[1]]} {gt_np[points[0]+3][points[1]]}")
                if replace_gt == 4:
                    bev_img[j][9 - i] = 255
                else:
                    bev_img[j][9 - i] = 100
            else:
                bev_img[j][9-i] = 100

    for i, line in enumerate(front_points_right):
        for j, points in enumerate(line):
            print((10+i, j), (points[0], points[1]), gt_np[points[0]][points[1]])
            if gt_np[points[0]][points[1]] == 4:
                bev_img[j][10+i] = 255
            elif gt_np[points[0]][points[1]] == 0:

                replace_gt = gt_np[points[0]+1][points[1]] or gt_np[points[0]+2][points[1]] or gt_np[points[0]+3][points[1]]
                print(f"replace_gt = {replace_gt} {gt_np[points[0]+1][points[1]]} {gt_np[points[0]+2][points[1]]} {gt_np[points[0]+3][points[1]]}")
                if replace_gt == 4:
                    bev_img[j][10+i] = 255
                else:
                    bev_img[j][10+i] = 100
            else:
                bev_img[j][10+i] = 100





    # back_lines  = []


    return bev_img

def make_bev_img_back(bev_img ,gt_np):

    # 64 * 512  mask
    # 思路
    # 1. [h] 只关注车身周围10m的情况 那么mask中 大概[-32:,:]是有效的
    # 2. [w] 其中256为正前方  0/512为正后方  128为左  384为右
    # 3. 这意味着 在bev图中 我如果想映射正前方偏右一点点的一条直线 那么在projmask中应该取 256偏右的一条斜线
    # 4. bev先预设20*20（格栅） 用20条线从256开始 10往左 10往右 生成正前方bev 用另外20条线从0/512开始 10往0-> 10往512-< 生成正后方bev

    # 1. 生成20 * 2条线
    #  斜率变化要根据projmask来 想象成走路时路沿距离自己的角度变换
    back_up_left_x         = [x for x in range(511, 461, -5)]
    back_down_left_x       = [x for x in range(511, 391, -12)]
    back_up_right_x        = [x for x in range(0, 50, 5)]
    back_down_right_x      = [x for x in range(0, 120, 12)]
    back_up_y              = 32
    back_down_y            = 43
    back_up_left_coor      = [(x, back_up_y) for x in back_up_left_x]
    back_down_left_coor    = [(x, back_down_y) for x in back_down_left_x]
    back_up_right_coor     = [(x, back_up_y) for x in back_up_right_x]
    back_down_right_coor   = [(x, back_down_y) for x in back_down_right_x]
    back_points_left       = []
    back_points_right      = []
    epoch = 0
    for up_coor, down_coor in zip(back_up_left_coor, back_down_left_coor):
        # print(up_coor, down_coor)
        lines_left = calculate_line_points_by_iter_lb(up_coor, down_coor)

        back_points_left.append(lines_left)
        epoch += 1

    for up_coor, down_coor in zip(back_up_right_coor, back_down_right_coor):
        # print(up_coor, down_coor)
        lines_right = calculate_line_points_by_iter_rb(up_coor, down_coor)

        back_points_right.append(lines_right)
        epoch += 1

    print(lines_left)
    print(lines_right)

    for i, line in enumerate(back_points_left):
        for j, points in enumerate(line):
            print((9 - i, j+9), (points[0], points[1]), gt_np[points[0]][points[1]])
            if gt_np[points[0]][points[1]] == 4:
                bev_img[j+9][9-i] = 255
            elif gt_np[points[0]][points[1]] == 0:

                replace_gt = gt_np[points[0]+1][points[1]] or gt_np[points[0]+2][points[1]] or gt_np[points[0]+3][points[1]]
                print(f"replace_gt = {replace_gt} {gt_np[points[0]+1][points[1]]} {gt_np[points[0]+2][points[1]]} {gt_np[points[0]+3][points[1]]}")
                if replace_gt == 4:
                    bev_img[j+9][9 - i] = 255
                else:
                    bev_img[j+9][9 - i] = 100
            else:
                bev_img[j+9][9-i] = 100

    for i, line in enumerate(back_points_right):
        for j, points in enumerate(line):
            print((10+i, j+9), (points[0], points[1]), gt_np[points[0]][points[1]])
            if gt_np[points[0]][points[1]] == 4:
                bev_img[j+9][10+i] = 255
            elif gt_np[points[0]][points[1]] == 0:

                replace_gt = gt_np[points[0]+1][points[1]] or gt_np[points[0]+2][points[1]] or gt_np[points[0]+3][points[1]]
                print(f"replace_gt = {replace_gt} {gt_np[points[0]+1][points[1]]} {gt_np[points[0]+2][points[1]]} {gt_np[points[0]+3][points[1]]}")
                if replace_gt == 4:
                    bev_img[j+9][10+i] = 255
                else:
                    bev_img[j+9][10+i] = 100
            else:
                bev_img[j+9][10+i] = 100





    # back_lines  = []


    return bev_img


class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.split = split
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.ignore_class = []

    # get the data
    from dataset.kitti.parser import Parser
    self.parser = Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
        torch.nn.Module.dump_patches = True
        if self.ARCH["train"]["pipeline"] == "hardnet":
            from modules.network.HarDNet import HarDNet
            self.model = HarDNet(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

        if self.ARCH["train"]["pipeline"] == "res":
            from modules.network.ResNet import ResNet_34
            self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

            def convert_relu_to_softplus(model, act):
                for child_name, child in model.named_children():
                    if isinstance(child, nn.LeakyReLU):
                        setattr(model, child_name, act)
                    else:
                        convert_relu_to_softplus(child, act)

            if self.ARCH["train"]["act"] == "Hardswish":
                convert_relu_to_softplus(self.model, nn.Hardswish())
            elif self.ARCH["train"]["act"] == "SiLU":
                convert_relu_to_softplus(self.model, nn.SiLU())

        if self.ARCH["train"]["pipeline"] == "fid":
            from modules.network.Fid import ResNet_34
            self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["act"] == "Hardswish":
                convert_relu_to_softplus(self.model, nn.Hardswish())
            elif self.ARCH["train"]["act"] == "SiLU":
                convert_relu_to_softplus(self.model, nn.SiLU())

#     print(self.model)
    w_dict = torch.load(modeldir + "/SENet_train_best",
                        map_location=lambda storage, loc: storage)
    self.model.load_state_dict(w_dict['state_dict'], strict=True)
    print(f"params: {sum(p.numel() for p in self.model.parameters())}")
    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())
    print(self.parser.get_n_classes())


    # iou
    self.evaluator = iouEval(self.parser.get_n_classes(),
                             self.device, self.ignore_class)

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    cnn = []
    knn = []
    self.evaluator.reset()

    if self.split == None:

        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)

        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)


    elif self.split == 'valid':
        self.infer_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    elif self.split == 'train':
        self.infer_subset(loader=self.parser.get_train_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    else:
        self.infer_subset(loader=self.parser.get_test_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)

    accuracy = self.evaluator.getacc()
    jaccard, class_jaccard = self.evaluator.getIoU()
    class_jaccard = list(np.array(class_jaccard.cpu()))[1:]
    print(f"ACC {accuracy} IOU {jaccard} classIOU {class_jaccard}")
    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return

  def infer_subset(self, loader, to_orig_fn,cnn,knn):
    # switch to evaluate mode

    self.model.eval()

    prunemodel = True
    if prunemodel:
        import torch.nn.utils.prune as prune
        parameters_to_prune = (
            (self.model.conv1.conv, 'weight'),
            (self.model.conv2.conv, 'weight'),
            (self.model.conv3.conv, 'weight'),
            (self.model.conv_1.conv, 'weight'),
            (self.model.conv_2.conv, 'weight'),
            # (self.model.layer1, 'weight'),
            # (self.model.layer2, 'weight'),
            # (self.model.layer3, 'weight'),
            # (self.model.layer4, 'weight'),
            (self.model.semantic_output, 'weight'),
        )
        # classIOU[0.9766440390326686, 0.0, 0.9738700564971752, 0.970984251413425, 0.9818743222235874]
        # 0.2 classIOU [0.9766773434019896, 0.0, 0.9738700564971752, 0.9709419544038237, 0.9818948200700914] time:0.04907272313092206
        # 0.5 classIOU [0.974993121256376, 0.0, 0.9721536834684525, 0.9706429517715703, 0.9810753032975256] time:0.05051762348896748
        # 0.9 classIOU [0.8323052179401046, 0.0, 0.6719492868462758, 0.8883645354998234, 0.8840020300010736]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.5)
        torch.save({'state_dict': self.model.state_dict()},
                   "test_prune0.5.pth")
    total_time=0
    total_frames=0
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      for i, (proj_in, proj_mask, proj_labels, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()
        end = time.time()

        if self.ARCH["train"]["aux_loss"]:
            with torch.cuda.amp.autocast(enabled=True):
                [proj_output, x_2, x_3, x_4] = self.model(proj_in)
        else:
            with torch.cuda.amp.autocast(enabled=True):
                proj_output = self.model(proj_in)

        proj_argmax = proj_output[0].argmax(dim=0)
        proj_labels = proj_labels.cuda().long()
        self.evaluator.addBatch(proj_argmax, proj_labels)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
              "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if self.post:
            # knn postproc
            unproj_argmax = self.post(proj_range,
                                      unproj_range,
                                      proj_argmax,
                                      p_x,
                                      p_y)
#             # nla postproc
#             proj_unfold_range, proj_unfold_pre = NN_filter(proj_range, proj_argmax)
#             proj_unfold_range=proj_unfold_range.cpu().numpy()
#             proj_unfold_pre=proj_unfold_pre.cpu().numpy()
#             unproj_range = unproj_range.cpu().numpy()
#             #  Check this part. Maybe not correct (Low speed caused by for loop)
#             #  Just simply change from
#             #  https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI/blob/7f90b45a765b8bba042b25f642cf12d8fccb5bc2/semantic_inference.py#L177-L202
#             for jj in range(len(p_x)):
#                 py, px = p_y[jj].cpu().numpy(), p_x[jj].cpu().numpy()
#                 if unproj_range[jj] == proj_range[py, px]:
#                     unproj_argmax = proj_argmax[py, px]
#                 else:
#                     potential_label = proj_unfold_pre[0, :, py, px]
#                     potential_range = proj_unfold_range[0, :, py, px]
#                     min_arg = np.argmin(abs(potential_range - unproj_range[jj]))
#                     unproj_argmax = potential_label[min_arg]

        else:
            # put in original pointcloud using indexes
            unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("KNN Infered seq", path_seq, "scan", path_name,
              "in", res, "sec")
        knn.append(res)
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        # pred_np.tofile(path)

        show_scans = True
        if show_scans:
            # get the first scan in batch and project points
            mask_np = proj_mask[0].cpu().numpy()
            depth_np = proj_in[0][0].cpu().numpy()
            pred_np = proj_argmax[0].cpu().numpy()
            gt_np = proj_labels[0].cpu().numpy()
            out = make_log_img(depth_np, mask_np, pred_np, gt_np, self.parser.to_color)
            name_proj = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", str(i) + "_proj.png")
            print(f"png path = {name_proj}")

            cv2.imwrite(name_proj, out)
            print(name_proj)
            bev_img = np.zeros((20, 20))
            bev = make_bev_img_front(bev_img, gt_np)
            bev = make_bev_img_back(bev_img, gt_np)

            name_bev = os.path.join(self.logdir, "sequences",
                                path_seq, "predictions", str(i) + "_bev.png")

            bev = cv2.resize(bev, (200, 200))
            bev = cv2.cvtColor(bev.astype("uint8"), cv2.COLOR_RGB2BGR)
            bev = cv2.rectangle(bev, (90,90), (100,100), (0,0,255), thickness=-1)
            cv2.imwrite(name_bev, bev)
