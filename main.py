import pandas

from util.Reader import read_event, sample_dis, sample_dis_dul
from util.drawPicture import draw_fixation, draw_multi_fiaxtion
from util.excelOperate import write_excel_xls

if __name__ == '__main__':

    eventFile = "data/zjb2_Events.txt"
    # sampleFile1 = "data/2022.10.29/hcy1-1_Samples.txt"
    # sampleFile2 = "data/zjb1_Samples.txt"
    # sampleFile3 = "data/hjn500-2_Samples.txt"

    df1 = read_event(eventFile)
    # df1.to_csv("csv_file/hcy_event1.csv")
    # df2 = sample_dis(sampleFile2)
    # df3 = sample_dis(sampleFile3)
    # df4 = sample_dis_dul([sampleFile1, sampleFile2, sampleFile3])
    # #
    # df1.to_csv("csv_file/hcy1.csv")
    # df2.to_csv("zjb1.csv")
    # df4.to_csv("sample3.csv")

    # df3 = pandas.read_csv("csv_file/hcy1.csv")
    # # 创建一个文件
    # book_name_xls = 'sample2.xls'
    # sheet_name_xls = 'xls格式测试表'
    # value_title = [
    #     ["x", "y", "total_duration", "max_duration", "mean_duration", "std_duration", "count_fixation", "number",
    #      "filename", "label"], ]
    # write_excel_xls(book_name_xls, sheet_name_xls, value_title)
    #
    # draw_multi_fiaxtion(df4)

    draw_fixation(df1)
