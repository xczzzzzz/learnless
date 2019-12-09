#!/usr/bin/python
# coding:utf-8
# --------------------------------------------------------
# Written by 朱晨曦
# --------------------------------------------------------
import os
import numpy as np
import time
import cv2
import copy
RGB_X = 760
RGB_Y = 480


def threeD2twoD(image_3D):
    N = np.shape(image_3D)
    x_meshgrid = np.linspace(0, N[1] - 1, N[1], dtype=np.int32)
    y_meshgrid = np.linspace(0, N[0] - 1, N[0], dtype=np.int32)
    z_meshgrid = np.linspace(-N[2] / 2, N[2] / 2,
                             N[2], dtype=np.float32) / N[2]
    [X, Y, Z] = np.meshgrid(x_meshgrid, y_meshgrid, z_meshgrid)
    Z = np.exp(-1j * Z * 1.65 * np.pi)
    image_2D = np.max(image_3D * Z, 2)
    maxtric_W = vcc(image_2D,'linear')
    return maxtric_W


def vcc(complexData2D, varargin='linear'):
    '''
    input:
    - complexData2D:  just what it says :-)
    - string (optional) that determines the scaling:
       no argument: linear intensity scaling
       's'        : root scaling
       'l'        : log scaling

     magnitude displays the intensity
     - black:        minimum intensity
     - full color:   maximum intensity

     color diplays the phase
     - blue:   -pi
     - green:  -pi/2
     - yellow = 0;
     - orange:  pi/2
     - purple:  pi
     '''
    whiteMask = np.double(np.uint8(np.isinf(complexData2D)))
    complexData2D[whiteMask == 1] = 0
    rgbPEAK = [6 * np.pi / 16.0, -2 * np.pi / 16.0, -0.55 * np.pi]
    rgbWIDTH = np.array([1.4, 1, 0.4])
    rgbMEAN = np.array([0.1, 0.1, 0.1])
    phase = np.angle(complexData2D)
    phaseColorMap = np.ones((RGB_X, RGB_Y, 3))
    rgbMAXminMIN = 2 * (1 - rgbMEAN)
    rgbMINIMUM = 2 * (rgbMEAN - 0.5)
    for i in range(3):
        if rgbWIDTH[i] > 1:
            pow = rgbWIDTH[i]
            peak = np.pi + rgbWIDTH[i]
            phaseColorMap[:, :, i] = 1 - rgbMAXminMIN[i] * \
                (np.cos(np.angle(np.exp(1j * (phase - peak)))) / 2 + 0.5)**pow
        else:
            pow = 1 / rgbWIDTH[i]
            peak = rgbPEAK[i]
            phaseColorMap[:, :, i] = rgbMINIMUM[
                i] + rgbMAXminMIN[i] * (np.cos(np.angle(np.exp(1j * (phase - peak)))) / 2 + 0.5) ** pow
    phaseColorMap = np.maximum(phaseColorMap, whiteMask[:,:,np.newaxis].repeat(3,axis=2))
    if varargin=='linear':
        magnIntensityMap = np.abs(complexData2D)
        magnIntensityMap = magnIntensityMap / np.max(magnIntensityMap[:])
    if varargin=='s':
        magnIntensityMap = np.sqrt(np.abs(complexData2D))
        magnIntensityMap = magnIntensityMap / np.max(magnIntensityMap[:])
    if varargin == 'l':
        magnIntensityMap = np.log(np.exp(1) + np.abs(complexData2D))
        magnIntensityMap = (magnIntensityMap - np.min(magnIntensityMap[:])) / (np.max(magnIntensityMap[:]) - np.min(
            magnIntensityMap[:]))
    magnIntensityMap = np.maximum(magnIntensityMap, whiteMask)
    magnIntensityMap = magnIntensityMap[:,:,np.newaxis].repeat(3,axis=2)
    vcc_rgb = phaseColorMap* magnIntensityMap
    return vcc_rgb


def color(dat_dir):
    filenames = os.listdir(dat_dir)
    for filename in filenames:
        if filename.endswith('dat'):
            file_dir = os.path.join(dat_dir, filename)
            data = np.fromfile(file_dir, dtype=np.float32).reshape((-1, 2))
            mean_data = np.sqrt(data[:, 0]**2 + data[:, 1]**2)
            # imgRes = mean_data.reshape(480,760,16)
            imgRes = np.zeros((480, 760, 16), dtype=np.float32)
            idx = 0
            for nz in range(16):
                for ny in range(760):
                    for nx in range(480):
                        imgRes[nx, ny, nz] = mean_data[idx]
                        idx = idx + 1
            minValue = np.min(mean_data)
            sortedA = np.sort(mean_data)
            maxV = np.mean(sortedA[len(sortedA) - 2500:len(sortedA)])
            maxValue = maxV
            imgRes[0, 0, :] = minValue
            imgRes[0, 1, :] = maxValue

            ImgR = np.zeros((760, 480, 16), dtype=np.float32)
            imgRes = (imgRes - minValue) / (maxValue - minValue)
            for depth in range(16):
                ImgR[:, :, depth] = np.abs(np.rot90(imgRes[:, :, depth], -1))
            MinV = 0.25
            MaxV = 1
            ImgR[ImgR <= MinV] = MinV
            ImgR[ImgR >= MaxV] = MaxV
            ImgR = (ImgR - MinV) / (MaxV - MinV)
            # for i in range(15):
            #     cv2.imshow('image1',ImgR[:,:,i])
            #     cv2.waitKey(0)
            for i in range(16):
                x = np.mean(ImgR[:,:,i])
                if x > 0.001:
                    k = i
                    break
            if  k<3 :
                temp_ImgR = copy.deepcopy(ImgR)
                m = 3-k
                for i in range(16-m,k,-1):
                    ImgR[:,:,i+m] = ImgR[:,:,i]
                for i in range(16-m+1,16):
                    ImgR[:,:,(i-16+m)] = temp_ImgR[:,:,i]
                for  i in range(k-1):
                    ImgR[:,:,i+m] = temp_ImgR[:,:,i]
            if k >3 :
                temp_ImgR = copy.deepcopy(ImgR)
                m = k-3
                for i in range(16):
                    ImgR[:,:,i-m] = ImgR[:,:,i]
                for i in range(3):
                    ImgR[:,:,i] = ImgR[:,:,i+m]
                for i in range(16-m+1,16):
                    ImgR[:,:,i] = temp_ImgR[:,:,i-16+m]
            final_matric = threeD2twoD(ImgR)
            b, g, r = cv2.split(final_matric)
            final_matric = cv2.merge([r,g,b])
            cv2.imshow('image2',final_matric)
            cv2.waitKey(0)
            final_matric = final_matric/final_matric.max()
            final_matric = final_matric*255
            final_matric = final_matric.astype(np.uint8)
            cv2.imwrite('123.jpg',final_matric)
            # for i in range(3):
            #     cv2.imshow('image1',final_matric[:,:,i])
            #     cv2.waitKey(0)


if __name__ == "__main__":
    time1 = time.time()
    color(dat_dir='/home/zcx/PycharmProjects/color')
    time2 = time.time()
    print time2 - time1
