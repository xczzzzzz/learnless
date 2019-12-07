# -*- coding:utf-8 -*-
# !/usr/bin/env python
import json
import cv2
# from labemlme import utils
import numpy as np
import glob
import PIL.Image
import os


class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path='./new.json'):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.categories.append(self.categorie())
        self.change_dict = {
            '1': 16,
            '2': 14,
            '3': 12,
            '4': 11,
            '5': 13,
            '6': 15,
            '11': 10,
            '12': 8,
            '13': 6,
            '14': 5,
            '15': 7,
            '16': 9,
            '18': 4,
            '19': 2,
            '20': 0,
            '21': 1,
            '22': 3}

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                temp_dict = {}
                self.images.append(self.image(data, num))

                for shapes in data['shapes']:
                    label = shapes['label'].split('_')[0]
                    if label in self.change_dict:
                        real_indx = self.change_dict[label]
                        points = shapes['points']
                        points[0].append(2)
                        temp_dict[real_indx] = points[0]
                temp_list = []
                for i in range(len(temp_dict)):
                    temp_list.extend(temp_dict[i])
                temp_list = list(map(int, temp_list))
                self.annotations.append(
                    self.annotation(temp_list, label, num))
                self.annID += 1
                print(json_file)

    def image(self, data, num):
        image = {}
        # img = utils.img_b64_to_array(data['imageData'])  # 解析原图片数据
        # img=io.imread(data['imagePath']) # 通过图片路径打开图片
        img = cv2.imread(
            os.path.join(
                '/home/zcx/Documents/labelme6-zhu',
                data['imagePath']),
            0)
        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        image['file_name'] = data['imagePath'].split('/')[-1]

        self.height = height
        self.width = width

        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = 'person'
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = 'person'
        categorie['keypoints'] = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"]
        categorie['skeleton'] = [
            [
                16, 14], [
                14, 12], [
                17, 15], [
                    15, 13], [
                        12, 13], [
                            6, 12], [
                                7, 13], [
                                    6, 7], [
                                        6, 8], [
                                            7, 9], [
                                                8, 10], [
                                                    9, 11], [
                                                        2, 3], [
                                                            1, 2], [
                                                                1, 3], [
                                                                    2, 4], [
                                                                        3, 5], [
                                                                            4, 6], [
                                                                                5, 7]]

        return categorie

    def annotation(self, keypoints, label, num):
        annotation = {}
        annotation['keypoints'] = keypoints
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        annotation['num_keypoints'] = 17
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        # annotation['bbox'] = list(map(float, self.getbbox(points)))
        #
        annotation['category_id'] = 1
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label[1] == categorie['name']:
                return categorie['id']
        return -1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  #
        # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(
            self.data_coco,
            open(
                self.save_json_path,
                'w'),
            indent=4)  # indent=4 更加美观显示


labelme_json = glob.glob('/home/zcx/Documents/labelme6-zhu/*.json')
# labelme_json=['./1.json']

labelme2coco(
    labelme_json,
    '/home/zcx/Documents/labelme6-zhu/person_keypoints_val2017.json')
