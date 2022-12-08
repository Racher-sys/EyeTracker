# -*- coding: utf-8 -*-
#
# This file is part of PyGaze - the open-source toolbox for eye tracking
#
#	PyGazeAnalyser is a Python module for easily analysing eye-tracking data
#	Copyright (C) 2014  Edwin S. Dalmaijer
#
#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <http://www.gnu.org/licenses/>

# Gaze Plotter
#
# Produces different kinds of plots that are generally used in eye movement
# research, e.g. heatmaps, scanpaths, and fixation locations as overlays of
# images.
#
# version 2 (02 Jul 2014)

__author__ = "Edwin Dalmaijer"

# native
import os
# external
import numpy
import matplotlib
import numpy as np
from matplotlib import pyplot, image, patches
from util.excelOperate import write_excel_xls_append
import math
from PIL import Image

# # # # #
# LOOK

# COLOURS
# all colours are from the Tango colourmap, see:
# http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines#Color_Palette
COLS = {"butter": ['#fce94f',
                   '#edd400',
                   '#c4a000'],
        "orange": ['#fcaf3e',
                   '#f57900',
                   '#ce5c00'],
        "chocolate": ['#e9b96e',
                      '#c17d11',
                      '#8f5902'],
        "chameleon": ['#8ae234',
                      '#73d216',
                      '#4e9a06'],
        "skyblue": ['#729fcf',
                    '#3465a4',
                    '#204a87'],
        "plum": ['#ad7fa8',
                 '#75507b',
                 '#5c3566'],
        "scarletred": ['#ef2929',
                       '#cc0000',
                       '#a40000'],
        "aluminium": ['#eeeeec',
                      '#d3d7cf',
                      '#babdb6',
                      '#888a85',
                      '#555753',
                      '#2e3436',
                      '#00BFFF',
                      ],
        }
# FONT
FONT = {'family': 'Ubuntu',
        'size': 12}
matplotlib.rc('font', **FONT)


# # # # #
# FUNCTIONS

def draw_fixations(fixations, dispsize, imagefile=None, durationsize=True, durationcolour=True, alpha=0.5,
                   savefilename=None):
    """Draws circles on the fixation locations, optionally on top of an image,
    with optional weigthing of the duration for circle size and colour

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    durationsize	-	Boolean indicating whether the fixation duration is
                    to be taken into account as a weight for the circle
                    size; longer duration = bigger (default = True)
    durationcolour	-	Boolean indicating whether the fixation duration is
                    to be taken into account as a weight for the circle
                    colour; longer duration = hotter (default = True)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    fixations
    """

    # FIXATIONS
    fix = parse_fixations(fixations)

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # CIRCLES
    # duration weigths
    if durationsize:
        siz = 1 * (fix['dur'] / 30.0)
    else:
        siz = 1 * numpy.median(fix['dur'] / 30.0)
    if durationcolour:
        col = fix['dur']
    else:
        col = COLS['chameleon'][2]
    # draw circles
    ax.scatter(fix['x'], fix['y'], s=siz, c=col, marker='o', cmap='jet', alpha=alpha, edgecolors='none')

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.

    arguments

    gazepoints		-	a list of gazepoint tuples (x, y, dur)

    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh / 2)
    ## 一个疑问，为什么热力图的大小要比原始显示图像大一点:很简单，不然如果注视点在图像的边缘上，那高斯矩阵不久溢出来了。
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)
    heatmap = numpy.zeros(heatmapsize, dtype=float)
    # heatmap = numpy.zeros(heatmapsize)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = int(strt + gazepoints[i][0] - int(gwh / 2))
        y = int(strt + gazepoints[i][1] - int(gwh / 2))

        ax.text(x, y, str(i), fontsize=30, color='#050505')
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
    # resize heatmap
    heatmap = heatmap[strt:int(dispsize[1]) + strt, strt:int(dispsize[0]) + strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def draw_heatmap1(fixations, dispsize, imagefile=None, alpha=0.5, savefilename=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    durationweight	-	Boolean indicating whether the fixation duration is
                    to be taken into account as a weight for the heatmap
                    intensity; longer duration = hotter (default = True)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # FIXATIONS
    fix = parse_fixations(fixations)

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = 150
    gsdwh = gwh / 6
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = gwh / 2
    heatmapsize = dispsize[1] + 2 * strt, dispsize[0] + 2 * strt
    heatmap = numpy.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(fix['dur'])):
        # get x and y coordinates
        # x and y - indexes of heatmap array. must be integers
        x = strt + int(fix['x'][i]) - int(gwh / 2)
        y = strt + int(fix['y'][i]) - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh];
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * fix['dur'][i]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus * fix['dur'][i]
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def draw_raw(x, y, dispsize, imagefile=None, savefilename=None):
    """Draws the raw x and y data

    arguments

    x			-	a list of x coordinates of all samples that are to
                    be plotted
    y			-	a list of y coordinates of all samples that are to
                    be plotted
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    fixations
    """

    # image
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # plot raw data points
    ax.plot(x, y, 'o', color="white", markeredgecolor=COLS['aluminium'][5], markersize=10)

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def adjust(point, boxes):
    # 找出离point最近的boxes,更新point
    min_dis = 100000
    point_new = np.zeros(2)
    for i in range(len(boxes)):
        distance = math.sqrt(pow((point[0] - boxes[i][0]), 2) + pow((point[1] - boxes[i][1]), 2))
        if min_dis > distance:
            min_dis = distance
            tmp = i
    point_new[0] = boxes[tmp][0]
    point_new[1] = boxes[tmp][1]

    return point_new


def draw_adjust_rect(point, boxes, dispsize, imagefile=None, savefilename=None):
    """
    :param point: feature[[...],[...]]
    :param boxes: [[xmin,ymin,width,height]]
    :param dispsize:
    :param imagefile:
    :param savefilename:
    :return:
    """
    # 图像的长宽
    width_fig = 1280
    height_fig = 1024
    # 图片的长宽
    width_pic = 768
    height_pic = 614
    # 图像与图片之差
    width_adj = (width_fig - width_pic) / 2
    height_adj = (height_fig - height_pic) / 2

    center = []

    # 计算每一个box的中心点坐标
    for i in range(len(boxes)):
        a1 = int(boxes[i][0] + boxes[i][2] / 2 + width_adj)
        a2 = int(boxes[i][1] + boxes[i][3] / 2 + height_adj)
        center.append([a1, a2])

    width = 60
    height = 100
    excelPath = 'sample2.xls'
    # image
    fig, ax = draw_display(dispsize, imagefile=imagefile)
    # point1 = np.array(point1)
    # 找出总注视时长最长的和注视点个数最多的框起来。
    max_totaldur = np.max(point, axis=0)[2]
    max_count = np.max(point, axis=0)[6]
    max_singledur = np.max(point, axis=0)[3]
    print(max_totaldur)
    print(max_count)
    for i in range(len(point)):

        if point[i][2] == max_totaldur:
            point_new = adjust(point[i], center)
            ax.add_patch(
                patches.Rectangle((point_new[0] - width / 2, point_new[1] - height / 2), width, height, fill=False,
                                  color="purple", linewidth=2))
            ax.text(point_new[0] - width / 2 + 10, point_new[1] - height / 2 + 10, "max_totaldur", fontsize=15,
                    color='#050505')
        if point[i][3] == max_singledur:
            point_new = adjust(point[i], center)
            ax.add_patch(
                patches.Rectangle((point_new[0] - width / 2, point_new[1] - height / 2), width, height, fill=False,
                                  color="yellow", linewidth=2))
            ax.text(point_new[0] - width / 2 + 10, point_new[1] - height / 2 + 10, "max_totaldur", fontsize=15,
                    color='#050505')

        if point[i][6] == max_count:
            point_new = adjust(point[i], center)
            ax.add_patch(
                patches.Rectangle((point_new[0] - width / 2, point_new[1] - height / 2), width, height, fill=False,
                                  color="blue", linewidth=2))
            ax.text(point_new[0] - width / 2 + 10, point_new[1] - height / 2 + 10, "max_totaldur", fontsize=15,
                    color='#050505')

    ax.invert_yaxis()

    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def draw_aoi(point1, point2, dispsize, imagefile=None, savefilename=None):
    """
    :param point1: raw fixation:[x,y,dur]
    :param point2: aio center:[x, y, last, count_fixation]
    :param imagefile: background image
    :param savefilename:
    :return: fig
    """
    width = 50
    height = 50
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # 画出raw fixation
    for i in range(len(point1)):
        ax.plot(point1[i][0], point1[i][1], 'o', color=COLS['aluminium'][6], markeredgecolor=COLS['aluminium'][5],
                markersize=int(point1[i][2] / 12))
        ax.text(point1[i][0], point1[i][1], str(i))

    # max_count = np.max(point2, axis=0)[3]
    # flag = 0
    # if max_count > 1:
    #     flag = 1
    # 画出 aio center 两种选择,一种是画出最大的count的区域,一种是画出最后一个点的区域,但是优先画出最大数量的区域

    for i in range(len(point2)):
        color = "purple"
        if point2[i][2] == 1:
            color = "red"
        # if point2[i][3] == max_count and flag == 1:
        #     color = "red"
        # if i == len(point2)-1:
        #     color = "red"
        ax.add_patch(
            patches.Rectangle((point2[i][0] - width / 2, point2[i][1] - height / 2), width, height, fill=False,
                              color=color, linewidth=2))

    ax.invert_yaxis()

    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def draw_target(box, imagefile, savefilename):
    """

    :param box: [[xmin,ymin,width,height]]
    :param imagefile:
    :param savefilename:
    :return:
    """
    # 图像的长宽
    width_fig = 1280
    height_fig = 1024
    # 图片的长宽
    width_pic = 768
    height_pic = 614
    # 图像与图片之差
    width_adj = (width_fig - width_pic) / 2
    height_adj = (height_fig - height_pic) / 2

    # if not os.path.isfile(imagefile):
    #     raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
    #     # load image
    # img = image.imread(imagefile)
    # img2 = img[int(box[0]):int(box[0] + box[2]), int(box[1]):int(box[1] + box[3]), :]
    # image.imsave(img2, savefilename)


def draw_aoi_pedestrain(point1, point2, box, dispsize, imagefile=None, savefilename=None):
    """
    :param point1: raw fixation:[x,y,dur]
    :param point2: aio center:[x, y, last, count_fixation]
    :param box: [xmin, ymin, width, height]
    :param imagefile: background image
    :param savefilename:
    :return: fig
    """
    width = 50
    height = 50
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # 画出raw fixation
    for i in range(len(point1)):
        ax.plot(point1[i][0], point1[i][1], 'o', color=COLS['aluminium'][6], markeredgecolor=COLS['aluminium'][5],
                markersize=int(point1[i][2] / 12))
        ax.text(point1[i][0], point1[i][1], str(i))

    # max_count = np.max(point2, axis=0)[3]
    # flag = 0
    # if max_count > 1:
    #     flag = 1
    # 画出 aio center 两种选择,一种是画出最大的count的区域,一种是画出最后一个点的区域,但是优先画出最大数量的区域
    for i in range(len(point2)):
        color = "purple"
        if point2[i][2] == 1:
            color = "red"
        # if point2[i][3] == max_count and flag == 1:
        #     color = "red"
        # if i == len(point2)-1:
        #     color = "red"
        ax.add_patch(
            patches.Rectangle((point2[i][0] - width / 2, point2[i][1] - height / 2), width, height, fill=False,
                              color=color, linewidth=2))

    ax.add_patch(patches.Rectangle((box[0], box[1]), box[2], box[3], fill=False, color='green', linewidth=4))
    ax.invert_yaxis()

    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def draw_rect(point, dispsize, imagefile=None, savefilename=None):
    """
    :param point: list :[[x,y],...]此刻的point是一个feature
    :param point1: list :[[x,y,dur],...]
    :param dispsize: (width,height)
    :param imagefile:
    :param savefilename:
    :return:
    """
    width = 60
    height = 100
    excelPath = 'sample2.xls'
    # image
    fig, ax = draw_display(dispsize, imagefile=imagefile)
    # point1 = np.array(point1)
    # 找出总注视时长最长的和注视点个数最多的框起来。
    max_totaldur = np.max(point, axis=0)[2]
    max_count = np.max(point, axis=0)[6]
    max_singledur = np.max(point, axis=0)[3]
    print(max_totaldur)
    print(max_count)
    for i in range(len(point)):

        # ax.add_patch(patches.Rectangle((point[i][0] - width / 2, point[i][1] - height / 2), width, height, fill=False,
        #                                color="white", linewidth=1.5))
        # ax.text(point[i][0] - width / 2 + 10, point[i][1] - height / 2 + 10, str(i), fontsize=15, color='#050505')

        if point[i][2] == max_totaldur:
            ax.add_patch(
                patches.Rectangle((point[i][0] - width / 2, point[i][1] - height / 2), width, height, fill=False,
                                  color="purple", linewidth=2))
            ax.text(point[i][0] - width / 2 + 10, point[i][1] - height / 2 + 10, "max_totaldur", fontsize=15,
                    color='#050505')
        if point[i][3] == max_singledur:
            ax.add_patch(
                patches.Rectangle((point[i][0] - width / 2, point[i][1] - height / 2), width, height, fill=False,
                                  color="yellow", linewidth=2))
            ax.text(point[i][0] - width / 2 + 10, point[i][1] - height / 2 + 10, "max_singledur", fontsize=15,
                    color='#050505')

        if point[i][6] == max_count:
            ax.add_patch(
                patches.Rectangle((point[i][0] - width / 2, point[i][1] - height / 2), width, height, fill=False,
                                  color="blue", linewidth=2))
            ax.text(point[i][0] - width / 2 + 10, point[i][1] - height / 2 + 10, "max_count", fontsize=15,
                    color='#050505')

        # # 存储point
        # point[i].append(i)
        # point[i].append(savefilename)
        # write_excel_xls_append(excelPath, point[i])

    # 下面是画出矩阵顺便也将点也画出来，其实不用，因为我可以先画点，然后画出矩形框。
    # for j in range(point1.shape[0]):
    #     # ax.scatter(point1[j][0], point1[j][1], s=point1[j][2]/20, c='orange', edgecolor='None',alpha=0.6 )
    #     ax.add_patch(
    #         patches.Circle((point1[j][0], point1[j][1]), radius=point1[i][2] / 20, color='orange', edgecolor='black',
    #                        alpha=0.6))
    #     if j > 0:
    #         ax.plot([point1[j - 1][0], point1[j][0]], [point1[j - 1][1], point1[j][1]], color='orange', linewidth=3,
    #                 alpha=0.8)

    ax.invert_yaxis()

    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def draw_circle(point, dispsize, imagefile=None, savefilename=None):
    """
        :param point: list :[[x,y,dur],...]
        :param dispsize: (width,height)
        :param imagefile:
        :param savefilename:
        :return:
        """
    # image
    fig, ax = draw_display(dispsize, imagefile=imagefile)
    point = np.array(point)
    for i in range(point.shape[0]):
        ax.add_patch(
            patches.Circle((point[i][0], point[i][1]), radius=point[i][2] / 10, color='red', edgecolor='black',
                           alpha=0.6))
        if i > 0:
            ax.plot([point[i - 1][0], point[i][0]], [point[i - 1][1], point[i][1]], color='red', linewidth=2,
                    alpha=0.8)

    ax.invert_yaxis()

    if savefilename != None:
        fig.savefig(savefilename, format='jpg')

    return fig


def draw_multi_circle(point, dispsize, imagefile=None, savefilename=None):
    """
        :param point: list :[[x,y,dur],...]
        :param dispsize: (width,height)
        :param imagefile:
        :param savefilename:
        :return:
        """
    # image
    fig, ax = draw_display(dispsize, imagefile=imagefile)
    point = np.array(point)
    print(point.shape)
    color = ['orange', 'green', 'red']
    for j in range(point.shape[0]):
        for i in range(point[j].shape[0]):
            ax.add_patch(
                patches.Circle((point[j][i][0], point[j][i][1]), radius=point[j][i][2] / 10, color=color[j],
                               edgecolor='black',
                               alpha=0.6))
            if i > 0:
                ax.plot([point[j][i - 1][0], point[j][i][0]], [point[j][i - 1][1], point[j][i][1]], color=color[j],
                        linewidth=2,
                        alpha=0.8)

    ax.invert_yaxis()

    if savefilename != None:
        fig.savefig(savefilename, format='jpg')

    return fig


def draw_scanpath(fixations, saccades, dispsize, imagefile=None, alpha=0.5, savefilename=None):
    """Draws a scanpath: a series of arrows between numbered fixations,
    optionally drawn over an image

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']
    saccades		-	a list of saccade ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Esac']
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # image
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # FIXATIONS
    # parse fixations
    fix = parse_fixations(fixations)
    # draw fixations
    ax.scatter(fix['x'], fix['y'], s=(1 * fix['dur'] / 30.0), c=COLS['chameleon'][2], marker='o', cmap='jet',
               alpha=alpha, edgecolors='none')
    # draw annotations (fixation numbers)
    for i in range(len(fixations)):
        ax.annotate(str(i + 1), (fix['x'][i], fix['y'][i]), color=COLS['aluminium'][5], alpha=1,
                    horizontalalignment='center', verticalalignment='center', multialignment='center')

    # SACCADES
    if saccades:
        # loop through all saccades
        for st, et, dur, sx, sy, ex, ey in saccades:
            # draw an arrow between every saccade start and ending
            ax.arrow(sx, sy, ex - sx, ey - sy, alpha=alpha, fc=COLS['aluminium'][0], ec=COLS['aluminium'][5], fill=True,
                     shape='full', width=10, head_width=20, head_starts_at_zero=False, overhang=0)

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig


# # # # #
# HELPER FUNCTIONS


def draw_display(dispsize, imagefile=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it

    arguments

    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)

    returns
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                    with a size of dispsize, and an image drawn onto it
                    if an imagefile was passed
    """

    # construct screen (black background)
    _, ext = os.path.splitext(imagefile)
    ext = ext.lower()
    data_type = 'float32' if ext == '.png' else 'uint8'
    screen = numpy.zeros((dispsize[1], dispsize[0], 3), dtype=data_type)
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = image.imread(imagefile)
        # print(img.shape)
        # flip image over the horizontal axis
        # (do not do so on Windows, as the image appears to be loaded with
        # the correct side up there; what's up with that? :/)
        if not os.name == 'nt':
            img = numpy.flipud(img)
        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = int(dispsize[0] / 2 - w / 2)
        y = int(dispsize[1] / 2 - h / 2)
        # draw the image on the screen
        screen[y:y + h, x:x + w, :] += img
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)  # , origin='upper')

    return fig, ax


def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = numpy.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = numpy.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M


def parse_fixations(fixations):
    """Returns all relevant data from a list of fixation ending events

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']

    returns

    fix		-	a dict with three keys: 'x', 'y', and 'dur' (each contain
                a numpy array) for the x and y coordinates and duration of
                each fixation
    """

    # empty arrays to contain fixation coordinates
    fix = {'x': numpy.zeros(len(fixations)),
           'y': numpy.zeros(len(fixations)),
           'dur': numpy.zeros(len(fixations))}
    # get all fixation coordinates
    for fixnr in range(len(fixations)):
        ex, ey, dur = fixations[fixnr]
        fix['x'][fixnr] = ex
        fix['y'][fixnr] = ey
        fix['dur'][fixnr] = dur

    return fix
