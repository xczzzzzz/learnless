#!/usr/bin/python
# coding:utf-8
# --------------------------------------------------------
# Written by 朱晨曦
# --------------------------------------------------------
import numpy as np


def voting_system(result1, result2):
    np_box1 = np.asanyarray(result1['box'], dtype=np.float32)
    np_box2 = np.asarray(result2['box'], dtype=np.float32)
    max_iou, _, _ = iou_calc(np_box1, np_box2)
    before_max_iou = max_iou.max(axis=1)
    after_max_iou = max_iou.max(axis=0)
    before_iou_keep = (max_iou == before_max_iou[:, np.newaxis])
    after_iou_keep = (max_iou == after_max_iou[np.newaxis, :])
    iou_keep = before_iou_keep & after_iou_keep
    matrix_iou_keep = max_iou > 0.8
    class_1 = np.asanyarray(result1['class'], dtype=np.float32)
    class_2 = np.asanyarray(result2['class'], dtype=np.float32)
    matrix_class_keep = class_1[:, np.newaxis] == class_2[np.newaxis, :]
    keep = iou_keep & matrix_iou_keep
    keep2 = keep & matrix_class_keep
    voting_matrix = keep ^ keep2
    indx = np.where(voting_matrix)
    if result1['score'][int(indx[0])] < result2['score'][int(indx[1])]:
        result1['class'][int(indx[0])] = result2['class'][int(indx[1])]
    else:
        result2['class'][int(indx[1])] = result1['class'][int(indx[0])]
    class_1 = np.asanyarray(result1['class'], dtype=np.float32)
    class_2 = np.asanyarray(result2['class'], dtype=np.float32)
    matrix_class_keep = class_1[:, np.newaxis] == class_2[np.newaxis, :]
    keep = iou_keep & matrix_iou_keep
    keep2 = keep & matrix_class_keep
    before_keep = np.array(-keep2.sum(axis=1) + 1, dtype='bool')
    after_keep = np.array(-keep2.sum(axis=0) + 1, dtype='bool')
    adding_box = np.where(after_keep)
    for indx in adding_box:
        for i in indx:
            if float(result2['score'][i]) > 0.9:
                result1['score'].append(result2['score'][i])
                result1['class'].append(result2['class'][i])
                result1['box'].append(result2['box'][i])

    return result1


def iou_calc(det_boxes, filter_boxes):
    det_y1 = det_boxes[:, 0]
    det_x1 = det_boxes[:, 1]
    det_y2 = det_boxes[:, 2]
    det_x2 = det_boxes[:, 3]
    det_areas = (det_x2 - det_x1 + 1) * (det_y2 - det_y1 + 1)

    filter_y1 = filter_boxes[:, 0]
    filter_x1 = filter_boxes[:, 1]
    filter_y2 = filter_boxes[:, 2]
    filter_x2 = filter_boxes[:, 3]
    filter_areas = (filter_x2 - filter_x1 + 1) * (filter_y2 - filter_y1 + 1)

    xx1 = np.maximum(det_x1[:, np.newaxis], filter_x1)
    xx2 = np.minimum(det_x2[:, np.newaxis], filter_x2)
    yy1 = np.maximum(det_y1[:, np.newaxis], filter_y1)
    yy2 = np.minimum(det_y2[:, np.newaxis], filter_y2)

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    inter = np.array(inter, dtype='float32')
    iou_by_all = inter / \
        (-inter + det_areas[:, np.newaxis] + filter_areas[np.newaxis, :])
    iou_by_det = inter / det_areas[:, np.newaxis]
    iou_by_filter = inter / filter_areas[np.newaxis, :]

    return iou_by_all, iou_by_det, iou_by_filter


if __name__ == '__main__':
    with open('p1.txt', 'r') as f1:
        lines1 = f1.readlines()
    with open('r1.txt', 'r') as f2:
        lines2 = f2.readlines()
        class_list = []
        box = []
        score = []
    for line in lines1:
        list = line.strip().split()
        if len(list) != 0:
            class_list.append(list[0])
            box.append([float(list[1].strip(',')),
                        float(list[2].strip(',')),
                        float(list[3].strip(',')),
                        float(list[4].strip(','))])
            score.append(list[5])
    result1 = dict(zip(['class', 'box', 'score'], [class_list, box, score]))
    class_list = []
    box = []
    score = []
    for line in lines2:
        list = line.strip().split()
        if len(list) != 0:
            class_list.append(list[0])
            box.append([float(list[1].strip(',')),
                        float(list[2].strip(',')),
                        float(list[3].strip(',')),
                        float(list[4].strip(','))])
            score.append(0.99)
    result2 = dict(zip(['class', 'box', 'score'], [class_list, box, score]))
    if len(result1['class']) > len(result2['class']):
        final = voting_system(result2, result1)
    else:
        final = voting_system(result1, result2)
    print final
