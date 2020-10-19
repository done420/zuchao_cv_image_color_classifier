# coding:utf-8
### package:
import extcolors,webcolors
import os,cv2,PIL,json
import numpy as np
from PIL import Image
import pandas as pd
from annoy import AnnoyIndex
from scipy.spatial import KDTree



########### 颜色配置信息

COLOR_COLLOCATION = {
	'BlueSeries': '蓝色系',
	'BrownSeries': '棕色系',
	'GraySeries': '灰色系',
	'GreenSeries': '绿色系',
	'OrangeSeries': '橙色系',
	'PinkSeries': '粉色系',
	'PurpleSeries': '紫色系',
	'RedSeries': '红色系',
	'YellowSeries': '黄色系'
}

jsonFile = '/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/image_color_classifier/colour_index.json'
IndexFile = '/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/image_color_classifier/colour_index.tree'


def color_to_name(requested_rgb_colour): #### 获取颜色名称（英文）
    #### 居其家居颜色匹配，获取颜色英文名称
    EN_NAMES, EN_COLOURS = [], []
    WEB_DICT = webcolors.CSS3_HEX_TO_NAMES #webcolors.css3_hex_to_names

    for hex_color,name in WEB_DICT.items():
        EN_NAMES.append(name)
        EN_COLOURS.append(webcolors.hex_to_rgb(hex_color))

    WB_SPACEDB = KDTree(EN_COLOURS)

    try:
        closest_en_name = webcolors.rgb_to_name(requested_rgb_colour)
    except ValueError:
        wb_dist, wb_index = WB_SPACEDB.query(requested_rgb_colour)
        closest_en_name = EN_NAMES[wb_index]
        # closest_wb_rgb = EN_COLOURS[wb_index]

    return closest_en_name



def load_json():
    color_stand =  [(ele["rgb_value"], ele["rbg_hex"], color_to_name(tuple(ele["rgb_value"])), ele["name_cn"]) for K, V in DICT.items() for ele in V['color_data']]
    df = pd.DataFrame(color_stand, columns=['rgb_value', 'rbg_hex', 'name_en', 'name_cn'])
    return df


def build_index(color_json,indexfile):
    DICT = json.load(open(color_json, "r"))
    color_stand =  [(ele["rgb_value"], ele["rbg_hex"], color_to_name(tuple(ele["rgb_value"])), ele["name_cn"]) for K, V in DICT.items() for ele in V['color_data']]
    df = pd.DataFrame(color_stand, columns=['rgb_value', 'rbg_hex', 'name_en', 'name_cn'])

    f = len(df['rgb_value'][0]) ### 特征长度，即rgb值,为3
    t = AnnoyIndex(f, metric='euclidean')
    n_tree = len(df['rgb_value']) #### 要建立的分类树，一般为物体的种类个数，为189

    for i, vector in enumerate(df['rgb_value']):
        t.add_item(i, vector)
    _ = t.build(n_tree)

    t.save(indexfile)#### 保存索引文件
    if os.path.exists(indexfile):
        print("index build: done.")


####### 创建索引
if not os.path.exists(IndexFile):
    build_index(jsonFile, IndexFile)


###### 颜色匹配
def color_map(rgb_vector):
    num_similar = 1
    ids = INDEXER.get_nns_by_vector(rgb_vector, num_similar)
    df_similar = DATA.iloc[ids]
    Lists = np.array(df_similar).tolist()[0]

    return Lists



##########
file = open(jsonFile, "r")
DICT = json.load(file)
file.close()

colorname2rgb = [(ele["name_cn"], ele["rgb_value"]) for K, V in DICT.items() for ele in V['color_data']]
Colorname2RGB = dict([ele[0], tuple(ele[1])] for ele in colorname2rgb)
color2collocation = [(ele["name_cn"], K) for K, V in DICT.items() for ele in V['color_data']]
Color2Collocation = dict([ele[0], ele[1]] for ele in color2collocation)

DATA = load_json()
INDEXER = AnnoyIndex(3, metric='euclidean')
INDEXER.load(IndexFile)


def image_color_extractor(cv_mat,k=5):

    img = Image.fromarray(cv2.cvtColor(cv_mat,cv2.COLOR_BGR2RGB))#### 错误，传入原图cv_mat是rgb_img[y1:y2, x1:x2]
    img = Image.fromarray(cv_mat)#### 错误，传入原图cv_mat是rgb_img[y1:y2, x1:x2]
    colors, pixel_count = extcolors.extract_from_image(img)

    def get_map_color(colors):
        sorted_rgb = [ele[0] for i,ele in enumerate(colors)]
        annoy_map = [color_map(list(c)) for c in sorted_rgb]
        annoy_map = [[tuple(ele[0]), ele[1], ele[2], ele[3]] for i,ele in enumerate(annoy_map)]

        color_fre = dict([i,ele[1]/sum([x[1] for x in colors])] for i,ele in enumerate(colors))
        #print(color_fre)

        tmp_list = []
        indx = []
        tmp_rgb_fre = {}
        for i,ele in enumerate(annoy_map):
            #print(ele)
            ix = annoy_map.index(ele)
            indx.append(ix)
            fre = color_fre.get(i)
            if ele not in tmp_list:
                tmp_list.append(ele)
                tmp_rgb_fre[ix] = fre
            else:
                tmp_rgb_fre[ix] = tmp_rgb_fre[ix] + fre

        annoy_map = tmp_list
        #print(tmp_rgb_fre)

        rgb_fre = [tmp_rgb_fre.get(i) for i in tmp_rgb_fre]
        rgb_standard = [ele[0] for ele in annoy_map] ### rgb list
        hex_standard = [ele[1] for ele in annoy_map]
        name_en = [ele[2] for ele in annoy_map]
        name_cn = [ele[3] for ele in annoy_map]

        return rgb_standard, hex_standard, name_en, name_cn, rgb_fre

    def warm_cold(colors):
        warm_colors = []
        cold_colors = []
        for c in colors:
            r, g, b = c[0]
            if b > r:#color = 'cold'
                cold_colors.append(c)
            else:#color = 'warm'
                warm_colors.append(c)
        warm_colors_sorted = sorted(warm_colors, key=lambda x: x[1], reverse=True)
        cold_colors_sorted = sorted(cold_colors, key=lambda x: x[1], reverse=True)
        return warm_colors_sorted, cold_colors_sorted


    # warms,colds = warm_cold(colors)
    # warm_rgbs,warm_hexs, warm_ne,warm_nc = get_map_color(warms)
    # cold_rgbs, cold_hexs, cold_ne, cold_nc = get_map_color(colds)


    rgb_list, hex_list, names_en, names_cn,rgb_fre = get_map_color(colors)


    Result = {
        "rgbs": rgb_list[:k] if len(rgb_list)>k else rgb_list,
        "hexs": hex_list[:k] if len(rgb_list)>k else hex_list,
        'names_en': names_en[:k] if len(rgb_list)>k else names_en,
        "names_cn": names_cn[:k] if len(rgb_list)>k else names_cn,
        "rgb_fre": rgb_fre[:k] if len(rgb_list)>k else rgb_fre
    }

    return Result



def test():
    imgfile = "../demo.jpg"
    im = cv2.imread(imgfile)[1:200,1:700]
    result = image_color_extractor(im,3)

    print(len(result["rgbs"]))
    print(result)

    # color_name = dict([ele,result[ele] ] for ele in result if ele == 'names_en' or ele == 'names_cn')
    color_name = dict([ele,result[ele] ] for ele in result if ele == 'names_en' or ele == 'names_cn')
    print(color_name)

    #cv2.imwrite("./crop.png",im)





if __name__ == "__main__":
    test()