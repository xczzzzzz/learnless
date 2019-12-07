#!/usr/bin/python
#coding:utf-8
# --------------------------------------------------------
# Written by 朱晨曦
# --------------------------------------------------------
import json
json_file = '/home/zcx/PycharmProjects/color/person_keypoints_val2017.json'
val=json.load(open(json_file, 'r'))
json_file = '/home/zcx/Documents/labelme6-zhu/person_keypoints_val2017.json'
val2=json.load(open(json_file, 'r'))
k=1
