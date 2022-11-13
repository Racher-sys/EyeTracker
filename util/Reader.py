import pandas as pd
import numpy as np
from util.disperision import Dispersion


# 读取sample比较完整的文件
def read_sample(filename):
    f1 = open(filename, 'r')
    raw = f1.readlines()
    f1.close()
    data = {'GazePoint X': 0, 'GazePoint Y': 0, 'Pupil L': 0, 'Pupil R': 0, 'Image Path': 0, 'count': 0}
    count = -1
    k = 0
    df = pd.DataFrame(columns=['GazePoint X', 'GazePoint Y', 'Pupil L', 'Pupil R', 'Image Path', 'count'])
    flag = 0
    s_flag = 0
    c = 0
    # msg1 = 0
    for i in range(len(raw)):
        line = raw[i].replace('\n', '').replace('\r', '').split('\t')
        if '##' in line[0]:
            continue
        if 'MSG' in line[1]:
            msg = line[3][11:]
            if 'sxh' in msg:
                msg1 = msg
                flag = 1
                ## 表示可以开始存储数据了
                count += 1
                s_flag = 1
            else:
                flag = 0

        # 往后面继续存30个点看看
        if 'SMP' in line[1] and flag == 1:
            if float(line[21]) != 0 and float(line[22]) != 0 and c < 30:
                data['GazePoint X'] = float(line[21])
                data['GazePoint Y'] = float(line[22])
                data['Pupil L'] = float(line[9])
                data['Pupil R'] = float(line[12])
                data['Image Path'] = msg
                data['count'] = int(count)
                df.loc[k] = data

                k += 1
            s_flag = 1
            c = 0
        elif 'SMP' in line[1] and s_flag == 1:
            if c < 20:
                if float(line[21]) != 0 and float(line[22]) != 0:
                    data['GazePoint X'] = float(line[21])
                    data['GazePoint Y'] = float(line[22])
                    data['Pupil L'] = float(line[9])
                    data['Pupil R'] = float(line[12])
                    data['Image Path'] = msg1
                    data['count'] = int(count)
                    df.loc[k] = data
                    k += 1
                    c += 1
            else:
                s_flag = 0

    return df


# 读取sample不怎么完整的文件
def read_sample1(filename):
    f1 = open(filename, 'r')
    raw = f1.readlines()
    f1.close()
    data = {'GazePoint X': 0, 'GazePoint Y': 0, 'Pupil L': 0, 'Pupil R': 0, 'Image Path': 0, 'count': 0}
    count = -1
    k = 0
    df = pd.DataFrame(columns=['GazePoint X', 'GazePoint Y', 'Pupil L', 'Pupil R', 'Image Path', 'count'])
    flag = 0
    s_flag = 0
    c = 0
    # msg1 = 0
    for i in range(len(raw)):
        line = raw[i].replace('\n', '').replace('\r', '').split('\t')
        if '##' in line[0]:
            continue
        if 'MSG' in line[1]:
            msg = line[3][11:]
            if 'sxh' in msg:
                msg1 = msg
                flag = 1
                ## 表示可以开始存储数据了
                count += 1
                s_flag = 1
            else:
                flag = 0

        # 往后面继续存30个点看看
        if 'SMP' in line[1] and flag == 1:
            if float(line[13]) != 0 and float(line[14]) != 0 and c < 30:
                data['GazePoint X'] = float(line[13])
                data['GazePoint Y'] = float(line[14])
                data['Pupil L'] = float(line[9])
                data['Pupil R'] = float(line[12])
                data['Image Path'] = msg
                data['count'] = int(count)
                df.loc[k] = data

                k += 1
            s_flag = 1
            c = 0
        elif 'SMP' in line[1] and s_flag == 1:
            if c < 20:
                if float(line[13]) != 0 and float(line[14]) != 0:
                    data['GazePoint X'] = float(line[13])
                    data['GazePoint Y'] = float(line[14])
                    data['Pupil L'] = float(line[9])
                    data['Pupil R'] = float(line[12])
                    data['Image Path'] = msg1
                    data['count'] = int(count)
                    df.loc[k] = data
                    k += 1
                    c += 1
            else:
                s_flag = 0

    return df


def read_sample2(filename):
    f1 = open(filename, 'r')
    raw = f1.readlines()
    f1.close()
    data = {'GazePoint X': 0, 'GazePoint Y': 0, 'Pupil L': 0, 'Pupil R': 0, 'Image Path': 0, 'count': 0}
    count = -1
    k = 0
    df = pd.DataFrame(columns=['GazePoint X', 'GazePoint Y', 'Pupil L', 'Pupil R', 'Image Path', 'count'])
    flag = 0
    for i in range(len(raw)):
        line = raw[i].replace('\n', '').replace('\r', '').split('\t')
        if '##' in line[0]:
            continue
        if 'MSG' in line[1]:
            msg = line[3][11:]
            if 'sxh' in msg:
                flag = 1
                count += 1
            else:
                flag = 0

        if 'SMP' in line[1] and flag == 1:
            if float(line[13]) != 0 and float(line[14]) != 0:
                data['GazePoint X'] = float(line[13])
                data['GazePoint Y'] = float(line[14])
                data['Pupil L'] = float(line[9])
                data['Pupil R'] = float(line[12])
                data['Image Path'] = msg
                data['count'] = int(count)
                df.loc[k] = data
                k += 1

            # if float(line[21]) != 0 and float(line[22]) != 0:
            #     data['GazePoint X'] = float(line[21])
            #     data['GazePoint Y'] = float(line[22])
            #     data['Pupil L'] = float(line[9])
            #     data['Pupil R'] = float(line[12])
            #     data['Image Path'] = msg
            #     data['count'] = int(count)
            #     df.loc[k] = data
            #     k += 1
    return df


def read_event(filename):
    '''
    :param filename: raw event_file
    :return: dataframe struct including every fixation location
    This function is to read the fixation point of one block
    '''
    f1 = open(filename, 'r')
    raw = f1.readlines()
    f1.close()
    data = {'Location X': 0, 'Location Y': 0, 'Duration': 0, 'Image Path': 0, 'Mark': 0}
    flag = 0
    count = -1
    k = 0
    df = pd.DataFrame(columns=['Location X', 'Location Y', 'Duration', 'Image Path', 'Mark'], index=[])

    for i in range(len(raw)):
        line = raw[i].replace('\n', '').replace('\r', '').split('\t')
        if 'UserEvent' in line[0]:
            msg = line[4][11:]
            if 'sxh' in msg:
                flag = 1
                count += 1
            else:
                flag = 0

        if flag == 1:
            # 寻找fixation数据(注意这边需不需要加上散度)
            if 'Fixation L' in line[0]:
                data['Location X'] = float(line[6])
                data['Location Y'] = float(line[7])
                data['Duration'] = float(line[5]) / 1000
                data['Image Path'] = msg
                data['Mark'] = int(count)
                df.loc[k] = data
                k = k + 1

    print(df)
    return df


# 读取sample文件然后通过disperision生成fixation
def sample_dis(file):
    df = pd.DataFrame(
        columns=['Location X', 'Location Y', 'Duration', 'Image Path', 'Avg Pupil L', 'Avg Pupil R', 'Mark'], index=[])
    dat = {'Location X': 0, 'Location Y': 0, 'Duration': 0, 'Image Path': 0, 'Avg Pupil L': 0, 'Avg Pupil R': 0,
           'Mark': 0}
    data = read_sample2(file)
    tmp = 0
    F = []
    df_count = 0

    for i in range(data.shape[0]):
        # 获取mark
        mark = data.loc[i, 'count']
        if tmp == mark:
            x = data.loc[i, 'GazePoint X']
            y = data.loc[i, 'GazePoint Y']
            backgroundImg = data.loc[i, 'Image Path']
            pupill = data.loc[i, 'Pupil L']
            pupilr = data.loc[i, 'Pupil R']
            F.append([x, y, pupill, pupilr])
        else:
            # 直到下一个mark
            # 提取注视点
            F = np.array(F)
            d = Dispersion(F, 10, 100, 250)
            a = d.dispersion()

            for j in range(a.shape[0]):
                dat['Location X'] = a.loc[j, 'x']
                dat['Location Y'] = a.loc[j, 'y']
                dat['Duration'] = a.loc[j, 'duration']
                dat['Avg Pupil L'] = a.loc[j, 'avg pupill']
                dat['Avg Pupil R'] = a.loc[j, 'avg pupilr']
                dat['Image Path'] = backgroundImg
                dat['Mark'] = tmp
                df.loc[df_count] = dat
                df_count += 1
            F = []
            tmp = mark
    print(df)
    return df


def sample_dis_dul(files):
    """
    :param files: including many file
    :return:
    """
    df = pd.DataFrame(
        columns=['Location X', 'Location Y', 'Duration', 'Image Path', 'Avg Pupil L', 'Avg Pupil R', 'Mark', 'sub',
                 'order'],
        index=[])
    dat = {'Location X': 0, 'Location Y': 0, 'Duration': 0, 'Image Path': 0, 'Avg Pupil L': 0, 'Avg Pupil R': 0,
           'Mark': 0, 'sub': 0, 'order': 0}
    sub = 0
    df_count = 0
    for k in range(len(files)):
        print("---------------------------")
        data = read_sample1(files[k])
        tmp = 0
        order = 0
        F = []
        for i in range(data.shape[0]):
            ## 获取mark
            mark = data.loc[i, 'count']
            if tmp == mark:
                x = data.loc[i, 'GazePoint X']
                y = data.loc[i, 'GazePoint Y']
                backgroundImg = data.loc[i, 'Image Path']
                pupill = data.loc[i, 'Pupil L']
                pupilr = data.loc[i, 'Pupil R']
                F.append([x, y, pupill, pupilr])
            else:
                # 直到下一个mark
                # 提取注视点
                F = np.array(F)
                d = Dispersion(F, 10, 100, 250)
                a = d.dispersion()
                for j in range(a.shape[0]):
                    dat['Location X'] = a.loc[j, 'x']
                    dat['Location Y'] = a.loc[j, 'y']
                    dat['Duration'] = a.loc[j, 'duration']
                    dat['Avg Pupil L'] = a.loc[j, 'avg pupill']
                    dat['Avg Pupil R'] = a.loc[j, 'avg pupilr']
                    dat['Image Path'] = backgroundImg
                    dat['Mark'] = tmp
                    dat['sub'] = sub
                    dat['order'] = order
                    df.loc[df_count] = dat
                    df_count += 1
                    order += 1
                F = []
                tmp = mark
                order = 0
        sub += 1
    df.sort_values(by=['Image Path', 'sub', 'order'], inplace=True, ignore_index=True)

    return df
