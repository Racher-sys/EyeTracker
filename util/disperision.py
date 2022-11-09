from unicodedata import name
import numpy as np
import pandas as pd


class Dispersion():
    """Simple dispersion-based algorithm. 
	   
	   This is based on I-DT of Salvucci and Goldberg (2000).

	   Parameters:
        data：array of gaze point(sample,4)(x,y,pupill,pupilr)
		windowSize: the size of the window (in samples).
		threshold (pixels) the radius in which to consider a fixation.
	"""
    # 如果采样率正确那么设置一个时间窗口为（36ms）,但是有这么一个前提采样率正确
    def __init__(self, data, windowSize, threshold, sampleRate) -> None:
        self.windowSize = windowSize
        self.threshold = threshold
        self.data = data
        self.sampleRate = sampleRate

    def dispersion(self):

        df = pd.DataFrame(columns=['x', 'y', 'duration', 'avg pupill', 'avg pupilr'])
        df1 = pd.DataFrame(columns=['x', 'y'])

        a = -1

        fixation = {'x': 0, 'y': 0, 'duration': 0, 'avg pupill': 0, 'avg pupilr': 0}
        fix = {'x': 0, 'y': 0}

        window = []

        curWinSize = self.windowSize

        count = 0

        start = 0

        b = 0

        dataLen = self.data.shape[0]
        ## 初始化窗口的数
        # self.window = self.data[:curWinSize]

        while count < dataLen:

            while len(window) < curWinSize:
                if count >= dataLen:
                    break
                window.append(self.data[count])
                count += 1

            if len(window) < self.windowSize:
                break

            minx = miny = 3000
            maxx = maxy = 0

            for i in range(len(window)):
                for j in range(2):
                    if window[i][0] < minx:
                        minx = window[i][0]
                    if window[i][0] > maxx:
                        maxx = window[i][0]
                    if window[i][1] < miny:
                        miny = window[i][1]
                    if window[i][1] > maxy:
                        maxy = window[i][1]

            d = maxx - minx + maxy - miny

            if d <= self.threshold:
                a += 1
                while d <= self.threshold and count < dataLen:
                    window.append(self.data[count])
                    count += 1
                    if window[-1][0] < minx:
                        minx = window[-1][0]
                    if window[-1][0] > maxx:
                        maxx = window[-1][0]
                    if window[-1][1] < miny:
                        miny = window[-1][1]
                    if window[-1][1] > maxy:
                        maxy = window[-1][1]

                    d = maxx - minx + maxy - miny
                    curWinSize += 1
                fixation['x'] = np.mean(np.array(window), axis=0)[0]
                fixation['y'] = np.mean(np.array(window), axis=0)[1]
                # 这边的持续时间其实不能根据采样率计算，因为他的采样率是不一定的
                fixation['duration'] = curWinSize / self.sampleRate * 1000
                fixation['avg pupill'] = np.mean(np.array(window), axis=0)[2]
                fixation['avg pupilr'] = np.mean(np.array(window), axis=0)[3]

                df.loc[a] = fixation
                length = len(window)
                # 把一个fixation中的所有点都放在一起显示，而不是单独显示一个fixation点，它与raw点的区别是去除了几个非fixation的点
                for i in range(b, b + length):
                    fix['x'] = window[i - b][0]
                    fix['y'] = window[i - b][1]
                    df1.loc[i] = fix

                b = b + length
                start = count

            else:
                start += 1
                count = start
                window = []
                curWinSize = self.windowSize

        return df

# if __name__ == '__main__':
#
#     data = np.array([[863.81,384.13],[863.26,384.71],[863.78,385.71],[862.25,385.49],[862,385.47],[861.31,385.33],
#             [860.41,386.14],[860.86,387.36],[859.8,388.02],[860.72,389.47],[860,390.38],[860.31,391.76],
#             [860.66,392.34],[861.52,394.56],[862.29,395.68],[863.17,397.93],[863.91,398.96],[865.46,400.02],
#             [866.1,401.1], [864.57,400.25],[862.4,399.77],[861.38,398.98],[860.45,396.89],[861.25,397.27],
#             [859.35,395.09],[852.48,393.36],[851.47,392.36],[849.89,391.13],[848.76,389.9],[846.92,387.78],
#             [846.8,386.13],[845.08,383.11],[845.51,380.85],[846.09,380.64],[843.64,375.66],[845.04,376.66],
#             [846.82,375],[847.47,372.46],[848.72,370.78],[847.79,368.39],[847.84,367.39],[847.24,364.82],[848.23,364.79]]
#     )
#     d = Dispersion(data,25,35,250)
#
#     a = d.dispersion()

# print(a)
