import os.path

from util.gazeplotter import draw_raw, draw_rect, draw_heatmap
from util.disperision import Dispersion
from util.generateAoi import genterate_aoi, genterate_aoi_dul, genterate_aoi_dul_feature
from detection2 import load_image_into_numpy_array, process_image
import numpy as np
from PIL import Image


def draw_rawpoint(data):
    ## 读取某一行的数据
    tmp = 0
    x = []
    y = []
    for i in range(data.shape[0]):
        ## 获取mark
        mark = data.loc[i, 'count']
        if tmp == mark:
            x.append(data.loc[i, 'GazePoint X'])
            y.append(data.loc[i, 'GazePoint Y'])
            backgroundImg = data.loc[i, 'Image Path']
        else:
            ## 直到下一个mark
            # print("x:",x)
            # print("y:",y)
            draw_raw(x=x, y=y, dispsize=(1280, 1024), imagefile=backgroundImg, savefilename="img6/" + str(tmp))
            x = []
            y = []
            tmp = mark


def draw_rawpoint1(data):
    tmp = 0
    F = []
    for i in range(data.shape[0]):
        ## 获取mark
        mark = data.loc[i, 'count']
        if tmp == mark:
            x = data.loc[i, 'GazePoint X']
            y = data.loc[i, 'GazePoint Y']
            backgroundImg = data.loc[i, 'Image Path']
            F.append([x, y])
        else:
            # 直到下一个mark
            # 提取注视点
            F = np.array(F)
            d = Dispersion(F, 9, 35, 250)
            a = d.dispersion()
            # M1, M2 = genterate_aoi(a[['x','y']].values,40)
            M1, M2 = genterate_aoi(a[['x', 'y']].values, 40)
            # draw_raw(x = a['x'].values, y = a['y'].values, dispsize= (1280,1024), imagefile = backgroundImg, savefilename = "img2/"+str(tmp))
            draw_raw(M1, M2, dispsize=(1280, 1024), imagefile=backgroundImg,
                     savefilename="img8/" + str(tmp))
            # draw_rect(point=M,dispsize=(1280,1024),imagefile=backgroundImg,savefilename="img_test/"+str(tmp))
            F = []
            tmp = mark

def drawHeatMap(raw, output_name, background_image):
    display_width = 1280
    display_height = 1024

    alpha = 0.5
    ngaussian = 150
    sd = 0
    if len(raw[0]) == 2:
        gaze_data = list(map(lambda q: (int(q[0]), int(q[1]), 1), raw))
    else:
        gaze_data = list(map(lambda q: (int(q[0]), int(q[1]), int(q[2])), raw))

    draw_heatmap(gaze_data, (display_width, display_height), alpha=alpha, savefilename=output_name,
                 imagefile=background_image, gaussianwh=ngaussian, gaussiansd=None)



def draw_fixation(data):
    tmp = 0
    F = []
    X = []
    Y = []
    for i in range(data.shape[0]):
        ## 获取mark
        mark = data.loc[i, 'Mark']
        if tmp == mark:
            x = data.loc[i, 'Location X']
            y = data.loc[i, 'Location Y']
            dur = data.loc[i, 'Duration']
            backgroundImg = data.loc[i, 'Image Path']
            X.append(x)
            Y.append(y)
            F.append([x, y, dur])
        else:
            ## 直到下一个mark
            # 提取注视点
            F = np.array(F)
            print(F)
            drawHeatMap(raw=F, output_name=f"image/xm_temp/{str(tmp)}", background_image=backgroundImg)
            # print("F:", F)
            # M1, M2 = genterate_aoi(a[['x','y']].values,40)
            # feature = genterate_aoi(F, 90)
            # print(feature)

            # 检测行人，生成new_boxes.
            # image = Image.open(backgroundImg)
            # image_np = load_image_into_numpy_array(image)
            # image_process, new_boxes = process_image(image_np)

            # draw_raw(x=X, y=Y, dispsize=(1280, 1024), imagefile=backgroundImg,
            #          savefilename="image/hcy_temp/" + str(tmp))
            # # draw_raw(M1, M2, dispsize=(1280, 1024), imagefile=backgroundImg,
            #          savefilename="img8/" + str(tmp))

            # imagefile = "img4/" + backgroundImg[19:-4] + ".jpg"
            # if os.path.exists(imagefile):
            #     draw_circle(point=F, dispsize=(1280, 1024), imagefile=imagefile,
            #                 savefilename="img5/" + backgroundImg[19:-4] + ".jpg")
            # else:
            # draw_circle(point=feature, dispsize=(1280, 1024), imagefile="G:/eyetracker/"+backgroundImg,
            #             savefilename="zjb11_temp/" + backgroundImg[18:-4] + ".jpg")
            # if len(feature) != 0:
            # #     draw_adjust_rect(point=feature, boxes=new_boxes, dispsize=(1280, 1024),
            # #                      imagefile="image/hcy_temp/" + backgroundImg[19:-4] + ".jpg",
            # #                      savefilename="image/hcy_adj_judge/" + backgroundImg[19:-4] + ".jpg")
            #     draw_rect(point=feature, dispsize=(1280, 1024), imagefile=backgroundImg,
            #               savefilename="image/hcy_temp/" + backgroundImg[21:])
            # draw
            F = []
            X = []
            Y = []
            tmp = mark

            # 把tmp!=mark的存储下来
            x = data.loc[i, 'Location X']
            y = data.loc[i, 'Location Y']
            dur = data.loc[i, 'Duration']
            backgroundImg = data.loc[i, 'Image Path']
            X.append(x)
            Y.append(y)
            F.append([x, y, dur])


def draw_multi_fiaxtion(data):
    tmp = ""
    F1 = []
    F2 = []
    F3 = []
    for i in range(data.shape[0]):
        # 获取imgpath
        imgPath = data.loc[i, 'Image Path']
        if i == 0:
            tmp = imgPath
        if tmp == imgPath:
            x = data.loc[i, 'Location X']
            y = data.loc[i, 'Location Y']
            dur = data.loc[i, 'Duration']
            # pupil_l = data.loc[i, 'Avg Pupil L']
            # pupil_r = data.loc[i, 'Avg Pupil R']
            backgroundImg = data.loc[i, 'Image Path']
            sub = data.loc[i, 'sub']
            if sub == 0:
                F1.append([x, y, dur])
            elif sub == 1:
                F2.append([x, y, dur])
            elif sub == 2:
                F3.append([x, y, dur])

        else:
            ## 直到下一个image
            # 提取注视点
            F1 = np.array(F1)
            F2 = np.array(F2)
            F3 = np.array(F3)
            print(backgroundImg)

            # M = genterate_aoi_dul([F1, F2, F3], 90)

            feature = genterate_aoi_dul_feature([F1, F2, F3], 90)

            draw_rect(point=feature, dispsize=(1280, 1024), imagefile="img2_temp/" + backgroundImg[19:-4] + ".jpg",
                      savefilename="img2_judge/" + backgroundImg[19:-4] + ".jpg")

            # draw_multi_circle(point=[F1, F2, F3], dispsize=(1280, 1024), imagefile=backgroundImg,
            #                   savefilename="img3_temp/" + backgroundImg[19:-4] + ".jpg")
            F1 = []
            F2 = []
            F3 = []
            tmp = imgPath
