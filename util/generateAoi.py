import math
import numpy as np


# 注视区域的生成。
def genterate_aoi(fixData, threshod):
    """
    :param fixData: list [[x,y],...]
    :return:
    """
    length = len(fixData)
    distance = np.zeros((len(fixData), len(fixData)))
    flag = np.zeros(len(fixData))

    for i in range(length):
        for j in range(length):
            d = math.pow((math.pow(fixData[i][0] - fixData[j][0], 2) + math.pow(fixData[i][1] - fixData[j][1], 2)), 0.5)
            distance[i][j] = d
    M = []
    M1 = []
    M2 = []
    feature = []
    feature1 = []
    # 找到最后一个点出现在哪个区域
    last = 0
    for i in range(length):
        last = 0
        if flag[i] == 0:
            m = [fixData[i]]
            # 判断是否是最后一个点
            if i == length - 1:
                last = 1
            for j in range(length):
                if distance[i][j] < threshod and j != i:
                    m.append(fixData[j])
                    flag[j] = 1
                    if j == length - 1:
                        last = 1
            x = np.mean(np.array(m), axis=0)[0]
            y = np.mean(np.array(m), axis=0)[1]
            total_duration = np.sum(np.array(m), axis=0)[2]
            max_duration = np.max(np.array(m), axis=0)[2]
            mean_duration = np.mean(np.array(m), axis=0)[2]
            std_duration = np.std(np.array(m), axis=0)[2]
            count_fixation = len(m)

            M1.append(x)
            M2.append(y)
            M.append([x, y])
            feature.append([x, y, total_duration, max_duration, mean_duration, std_duration, count_fixation])

            feature1.append([x, y, last, count_fixation])
            # feature.append([x, y, total_duration, max_duration, mean_duration, count_duration])

    return feature1


def genterate_aoi_dul(fixDatas, threshod):
    """
    :param fixDatas: list [[x,y,duration],...][F1,F2,F3]
    :return:
    """
    length1 = fixDatas[0].shape[0]
    length2 = fixDatas[1].shape[0]
    length3 = fixDatas[2].shape[0]
    if length1 != 0:
        fixData = fixDatas[0]
        if length2 != 0:
            fixData = np.vstack((fixData, fixDatas[1]))
            if length3 != 0:
                fixData = np.vstack((fixData, fixDatas[2]))
        elif length3 != 0:
            fixData = np.vstack((fixData, fixDatas[2]))
    elif length2 != 0:
        fixData = fixDatas[1]
        if length3 != 0:
            fixData = np.vstack((fixData, fixDatas[2]))
    elif length3 != 0:
        fixData = fixDatas[2]
    else:
        return []

    length = fixData.shape[0]
    distance = np.zeros((length, length))
    flag = np.zeros(length)

    for i in range(length):
        for j in range(length):
            d = math.pow((math.pow(fixData[i][0] - fixData[j][0], 2) + math.pow(fixData[i][1] - fixData[j][1], 2)), 0.5)
            distance[i][j] = d

    M = []
    M1 = []
    M2 = []
    feature = []
    for i in range(length):
        if flag[i] == 0:
            m = [fixData[i]]
            for j in range(length):
                if distance[i][j] < threshod and j != i:
                    m.append(fixData[j])
                    flag[j] = 1
            x = np.mean(np.array(m), axis=0)[0]
            y = np.mean(np.array(m), axis=0)[1]
            total_duration = np.sum(np.array(m), axis=0)[2]
            max_duration = np.max(np.array(m), axis=0)[2]
            mean_duration = np.mean(np.array(m), axis=0)[2]
            std_duration = np.std(np.array(m), axis=0)[2]
            count_fixation = len(m)

            M1.append(x)
            M2.append(y)
            M.append([x, y])
            feature.append([x, y, total_duration, max_duration, mean_duration, std_duration, count_fixation])

            # feature.append([x, y, total_duration, max_duration, mean_duration, count_duration])

    return M


def genterate_aoi_dul_feature(fixDatas, threshod):
    """
    :param fixDatas: list [[x,y,duration],...][F1,F2,F3]
    :return:
    """
    length1 = fixDatas[0].shape[0]
    length2 = fixDatas[1].shape[0]
    length3 = fixDatas[2].shape[0]
    if length1 != 0:
        fixData = fixDatas[0]
        if length2 != 0:
            fixData = np.vstack((fixData, fixDatas[1]))
            if length3 != 0:
                fixData = np.vstack((fixData, fixDatas[2]))
        elif length3 != 0:
            fixData = np.vstack((fixData, fixDatas[2]))
    elif length2 != 0:
        fixData = fixDatas[1]
        if length3 != 0:
            fixData = np.vstack((fixData, fixDatas[2]))
    elif length3 != 0:
        fixData = fixDatas[2]
    else:
        return []

    length = fixData.shape[0]
    distance = np.zeros((length, length))
    flag = np.zeros(length)

    for i in range(length):
        for j in range(length):
            d = math.pow((math.pow(fixData[i][0] - fixData[j][0], 2) + math.pow(fixData[i][1] - fixData[j][1], 2)), 0.5)
            distance[i][j] = d

    M = []
    M1 = []
    M2 = []
    feature = []
    for i in range(length):
        if flag[i] == 0:
            m = [fixData[i]]
            for j in range(length):
                if distance[i][j] < threshod and j != i:
                    m.append(fixData[j])
                    flag[j] = 1
            x = np.mean(np.array(m), axis=0)[0]
            y = np.mean(np.array(m), axis=0)[1]
            total_duration = np.sum(np.array(m), axis=0)[2]
            max_duration = np.max(np.array(m), axis=0)[2]
            mean_duration = np.mean(np.array(m), axis=0)[2]
            std_duration = np.std(np.array(m), axis=0)[2]
            count_fixation = len(m)

            M1.append(x)
            M2.append(y)
            M.append([x, y])
            feature.append([x, y, total_duration, max_duration, mean_duration, std_duration, count_fixation])

    return feature
