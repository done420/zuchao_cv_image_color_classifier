# coding:utf-8
### package:

import os,glob,cv2,webcolors
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import KDTree

def mapping_to_closest_colour(requested_rgb_colour):
    ### rgb空间中的欧几里得距离进行匹配
    hexnames = webcolors.CSS3_HEX_TO_NAMES #webcolors.css3_hex_to_names
    names = []
    colors = []

    for hex, name in hexnames.items():
        names.append(name)
        colors.append(webcolors.hex_to_rgb(hex))
    spacedb = KDTree(colors)

    dist, index = spacedb.query(requested_rgb_colour)
    min_dist = dist
    closest_name = names[index]

    return min_dist, closest_name


def get_color_name(requested_rgb_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_rgb_colour)
        min_dist = None
    except ValueError:
        min_dist, closest_name = mapping_to_closest_colour(requested_rgb_colour)  # closest_colour(requested_rgb_colour)
        actual_name = None

    return closest_name, actual_name, min_dist



def get_image_main_color_to_color_name(image, number_of_colors = 3):
    """

    :param image:rgb image
    :param number_of_colors:k
    :return: closest_color_rgb(np.array, [175.2 178.0 187.6]) ,string type of color_name
    """
    resize_rate = 0.1
    w, h, c = image.shape
    H = int(resize_rate * h) if int(resize_rate * h) >0 else h
    W = int(resize_rate * w) if int(resize_rate * w) >0 else w
    modified_image = cv2.resize(image, (H, W), interpolation=cv2.INTER_AREA)
    # modified_image = image
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)


    try:
        clf = KMeans(n_clusters=number_of_colors)
        labels = clf.fit_predict(modified_image)
    except ValueError:
        clf = KMeans(n_clusters=1)
        labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    counts = dict(sorted(counts.items()))  ### 排序
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    rgb_clusters = ["{}".format(counts.get(i) / sum(counts.values())) for i in counts.keys()] ## 颜色聚类比例
    color_names = [get_color_name(tuple(map(int,i)))[0] for i in rgb_colors]

    closest_index = np.argmax(rgb_clusters)
    closest_color_name = color_names[closest_index] ## 最相似颜色名称
    closest_color_rgb = rgb_colors[closest_index] ## 最相似颜色rgb值


    return closest_color_rgb,closest_color_name

def test():
    # imgfile = "../demo.jpg"
    current_path = os.path.abspath(__file__) ## 当前目录
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    imgfile = os.path.join(father_path,'../demo.jpg')

    im = cv2.imread(imgfile)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    closest_color_rgb,closest_color_name = get_image_main_color_to_color_name(im)
    print("{}:\n predict_color: {}, rgb = {}".format(imgfile,closest_color_name,closest_color_rgb))

# if __name__ == "__main__":
#     test()

