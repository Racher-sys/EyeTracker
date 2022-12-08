import os.path

from util.gazeplotter import draw_raw, draw_rect, draw_heatmap, draw_aoi, draw_aoi_pedestrain, draw_target
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
            draw_raw(x=x, y=y, dispsize=(1280, 1024), imagefile=backgroundImg,
                     savefilename="image/raw/hcy_raw2/" + str(tmp))
            x = []
            y = []
            tmp = mark


# 画出没有黑边的方法
def draw_rawpoint_whole_img(data):
    ## 读取某一行的数据
    tmp = 0
    x = []
    y = []
    display_width = 1280
    display_height = 1024
    # 图片的长宽
    width_pic = 768
    height_pic = 614

    dispsize = (width_pic, height_pic)
    # 图像与图片之差
    width_adj = (display_width - width_pic) / 2
    height_adj = (display_height - height_pic) / 2

    for i in range(data.shape[0]):
        ## 获取mark
        mark = data.loc[i, 'count']
        if tmp == mark:
            x_temp = data.loc[i, 'GazePoint X'] - width_adj
            y_temp = data.loc[i, 'GazePoint Y'] - height_adj
            if x_temp >= 0 and x_temp <= width_pic and y_temp >= 0 and y_temp <= height_pic:
                x.append(x_temp)
                y.append(y_temp)
                backgroundImg = data.loc[i, 'Image Path']
        else:
            draw_raw(x=x, y=y, dispsize=dispsize, imagefile=backgroundImg,
                     savefilename="image/raw1/hjn_1000_raw/" + str(tmp))
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
        # 获取mark
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
            # 直到下一个mark
            # 提取注视点
            F = np.array(F)
            print(F)
            # drawHeatMap(raw=F, output_name=f"image/xm_temp/{str(tmp)}", background_image=backgroundImg)
            # print("F:", F)
            # M1, M2 = genterate_aoi(a[['x','y']].values,40)

            # 根据注视点生成注视区域

            feature = genterate_aoi(F, 50)
            print(feature)

            draw_aoi(point1=F, point2=feature, dispsize=(1280, 1024), imagefile=backgroundImg,
                     savefilename=f"image/aoi&target&seq/20221126/zjb6/{str(tmp)}")

            # 检测行人，生成new_boxes.
            # image = Image.open(backgroundImg)
            # image_np = load_image_into_numpy_array(image)
            # image_process, new_boxes = process_image(image_np)

            # draw_raw(x=X, y=Y, dispsize=(1280, 1024), imagefile=backgroundImg,
            #          savefilename="image/hcy_temp/" + str(tmp))
            # draw_raw(M1, M2, dispsize=(1280, 1024), imagefile=backgroundImg,
            #          savefilename="img8/" + str(tmp))

            # imagefile = "img4/" + backgroundImg[19:-4] + ".jpg"
            # if os.path.exists(imagefile):
            #     draw_circle(point=F, dispsize=(1280, 1024), imagefile=imagefile,
            #                 savefilename="img5/" + backgroundImg[19:-4] + ".jpg")
            # else:
            # draw_circle(point=feature, dispsize=(1280, 1024), imagefile="G:/eyetracker/"+backgroundImg,
            #             savefilename="zjb11_temp/" + backgroundImg[18:-4] + ".jpg")
            # if len(feature) != 0:
            #     draw_adjust_rect(point=feature, boxes=new_boxes, dispsize=(1280, 1024),
            #                      imagefile="image/hcy_temp/" + backgroundImg[19:-4] + ".jpg",
            #                      savefilename="image/hcy_adj_judge/" + backgroundImg[19:-4] + ".jpg")
            # draw_rect(point=feature, dispsize=(1280, 1024), imagefile=backgroundImg,
            #           savefilename="image/hcy_temp/" + backgroundImg[21:])
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


def judge(aoi, new_boxes):
    """
    :param aoi: aoi的坐标是感兴趣区域中心点的坐标[[x,y,last,count_fixation]]
    :param new_boxes: new_boxes的格式是[[xmin,ymin,width,height]]
    :return: 返回一个选择后的最终的boxes,因为在draw.rect的时候用的就是box的

    method :把感兴趣区域进行归类，归到某个行人身上。
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

    final_box = np.zeros(4)
    # sort有两行，第一行是每个行人框的感兴趣区域的个数，第二行是上一步中最有可能的目标感兴趣框所在的行人框上的标定。
    sort = np.zeros((2, len(new_boxes)))
    for i in range(len(aoi)):
        min = 1000
        record = 0
        for j in range(len(new_boxes)):
            # w = (xmin+width/2-x)
            w = abs(new_boxes[j][0] + new_boxes[j][2] / 2 + width_adj - aoi[i][0])
            # h = (ymin+height/2-y)
            h = abs(new_boxes[j][1] + new_boxes[j][3] / 2 + height_adj - aoi[i][1])
            if min > w:
                min = w
                record = j
            if min > h:
                min = h
                record = j
        sort[0][record] += 1
        sort[1][record] = aoi[i][2]

    print(sort)
    # 找出感兴趣区域个数最大的行人框，以及目标感兴趣所在的行人框。

    max_number = np.max(sort, axis=1)[0]
    # 优先选择max count 的行人框 ，如果有多个行人框中的max count是一样的，那么就选择有目标框的那个行人框。
    for i in range(len(new_boxes)):
        if sort[1][i] == 1:
            final_box = new_boxes[i]

    print(np.array(final_box))
    final_box = np.array(final_box)
    final_box1 = final_box
    # 调整宽度
    final_box[0] += width_adj
    final_box[1] += height_adj

    return final_box, final_box1


def draw_fixation1(data):
    tmp = 0
    F = []
    X = []
    Y = []
    for i in range(data.shape[0]):
        # 获取mark
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
            # 直到下一个mark
            # 提取注视点
            F = np.array(F)
            print(F)

            # 根据注视点生成注视区域

            feature = genterate_aoi(F, 50)
            print(feature)

            # draw_aoi(point1=F, point2=feature, dispsize=(1280, 1024), imagefile=backgroundImg,
            #          savefilename=f"image/aoi&target&count/zjb_target2/{str(tmp)}")

            # 检测行人，生成new_boxes.
            image = Image.open(backgroundImg)
            image_np = load_image_into_numpy_array(image)
            image_process, new_boxes = process_image(image_np)

            box, box1 = judge(feature, new_boxes)

            draw_target(box=box1, imagefile=backgroundImg, savefilename=f"image/crop/hcy1/{str(tmp)}.png")

            draw_aoi_pedestrain(point1=F, point2=feature, box=box, dispsize=(1280, 1024), imagefile=backgroundImg,
                                savefilename=f"image/aoi&target&ped/20221120/hcy1/{str(tmp)}")

            # 对于每一个new_box与aoi结合判断

            # draw_raw(x=X, y=Y, dispsize=(1280, 1024), imagefile=backgroundImg,
            #          savefilename="image/hcy_temp/" + str(tmp))
            # draw_raw(M1, M2, dispsize=(1280, 1024), imagefile=backgroundImg,
            #          savefilename="img8/" + str(tmp))

            # imagefile = "img4/" + backgroundImg[19:-4] + ".jpg"
            # if os.path.exists(imagefile):
            #     draw_circle(point=F, dispsize=(1280, 1024), imagefile=imagefile,
            #                 savefilename="img5/" + backgroundImg[19:-4] + ".jpg")
            # else:
            # draw_circle(point=feature, dispsize=(1280, 1024), imagefile="G:/eyetracker/"+backgroundImg,
            #             savefilename="zjb11_temp/" + backgroundImg[18:-4] + ".jpg")
            # if len(feature) != 0:
            #     draw_adjust_rect(point=feature, boxes=new_boxes, dispsize=(1280, 1024),
            #                      imagefile="image/hcy_temp/" + backgroundImg[19:-4] + ".jpg",
            #                      savefilename="image/hcy_adj_judge/" + backgroundImg[19:-4] + ".jpg")
            # draw_rect(point=feature, dispsize=(1280, 1024), imagefile=backgroundImg,
            #           savefilename="image/hcy_temp/" + backgroundImg[21:])
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
            # 直到下一个image
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
