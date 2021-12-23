# -*- coding: utf-8 -*-
# Author : Alven.Gu
import time
import jieba_fast as jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
from multiprocessing import Pool, cpu_count
import os

NUMBER_OF_PROCESSES = cpu_count()


def timmer(func):
    def deco(*args, **kwargs):
        print('\n函数：\033[32;1m{_funcname_}()\033[0m 开始运行：'.format(_funcname_=func.__name__))
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('函数: \033[32;1m{_funcname_}()\033[0m 运行了 {_time_}秒'
              .format(_funcname_=func.__name__, _time_=(end_time - start_time)))
        return res

    return deco


@timmer
def csv2txt(csv_file_path, txt_file_path):
    if not os.path.exists(txt_file_path):
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                with open(txt_file_path, 'a', encoding='utf-8-sig')as t:
                    for line in reader:
                        t.write('\r\n' + line[0])
        except Exception:
            with open(csv_file_path, 'r', encoding='gbk') as f:
                reader = csv.reader(f)
                with open(txt_file_path, 'a', encoding='utf-8-sig')as t:
                    for line in reader:
                        t.write('\r\n' + line[0])
    else:
        os.remove(txt_file_path)
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                with open(txt_file_path, 'a', encoding='utf-8-sig')as t:
                    for line in reader:
                        t.write('\r\n' + line[0])
        except Exception:
            with open(csv_file_path, 'r', encoding='gbk') as f:
                reader = csv.reader(f)
                with open(txt_file_path, 'a', encoding='utf-8-sig')as t:
                    for line in reader:
                        t.write('\r\n' + line[0])


@timmer
def file_cut(txt_file_path):
    with open(txt_file_path, 'r', encoding='utf-8-sig')as f:
        all_text_in_file = f.read().replace('\r', '').replace('\n', '').replace('\t', '')
        file_len = len(all_text_in_file)
        seg_len = int(file_len / NUMBER_OF_PROCESSES)
        str_list = [all_text_in_file[((i - 1) * seg_len):(i * seg_len - 1)] for i in
                    range(1, NUMBER_OF_PROCESSES + 1, 1)]
        return str_list


def chinese_word_segmentation_accurate_mode(file_string):
    word_generator = jieba.cut(file_string)
    word_dict = {}
    for word in word_generator:
        if len(word) < 1:
            continue
        else:
            word_dict[word] = word_dict.get(word, 0) + 1
    return word_dict


def chinese_word_segmentation_full_mode(file_string):
    word_generator = jieba.cut(file_string, cut_all=True)
    word_dict = {}
    for word in word_generator:
        if len(word) < 1:
            continue
        else:
            word_dict[word] = word_dict.get(word, 0) + 1
    return word_dict


def chinese_word_segmentation_search_mode(file_string):
    word_generator = jieba.cut_for_search(file_string)
    word_dict = {}
    for word in word_generator:
        if len(word) < 1:
            continue
        else:
            word_dict[word] = word_dict.get(word, 0) + 1
    return word_dict


@timmer
def dict2list(word_dict):
    word_list = list(word_dict.items())
    word_list.sort(key=lambda x: x[1], reverse=True)
    return word_list


@timmer
def chinese_word_segmentation(file_path, seg_mode='search'):
    if file_path[-4:] == '.csv':
        csv2txt(file_path, file_path[:-4] + '.txt')
        file_path = file_path[:-4] + '.txt'
    string_list = file_cut(file_path)
    pool = Pool(NUMBER_OF_PROCESSES)
    if seg_mode == 'accurate':
        dict_list = pool.map(chinese_word_segmentation_accurate_mode, string_list)
    elif seg_mode == 'full':
        dict_list = pool.map(chinese_word_segmentation_full_mode, string_list)
    else:
        dict_list = pool.map(chinese_word_segmentation_search_mode, string_list)
    pool.close()
    pool.join()
    word_dict = {}
    for dict_info in dict_list:
        word_dict = dict(word_dict, **dict_info)

    word_list = dict2list(word_dict)
    return word_list


@timmer
def generating_wordcloud(counts_dict, mask_image_path=None):
    if not mask_image_path:
        word_cloud = WordCloud(
            font_path="C:/Windows/Fonts/simfang.ttf",
            background_color="black", width=679, height=1019
        )
    else:
        cloud_mask = np.array(Image.open(mask_image_path))
        word_cloud = WordCloud(
            mask=cloud_mask,
            font_path="C:/Windows/Fonts/simfang.ttf",
            background_color="black", width=679, height=1019
        )
    word_cloud.generate_from_frequencies(counts_dict)
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def main():
    word_list = chinese_word_segmentation('1.txt', 'full')
    print(word_list)
    generating_wordcloud(dict(word_list[0:500]),'3.png')


if __name__ == '__main__':
    main()
    # coding=UTF-8