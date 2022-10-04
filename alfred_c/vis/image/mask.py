
"""
Draw masks on image,
every mask will has single id, and color are not same
also this will give options to draw detection or not
"""
import torch
import cv2
import numpy as np
from .common import get_unique_color_by_id, get_unique_color_by_id2, get_unique_color_by_id_with_dataset
from .det import draw_one_bbox
from PIL import Image
from .get_dataset_color_map import *
from .get_dataset_label_map import coco_label_map_list
import json

ALL_COLORS_MAP = {
    "cityscapes": create_cityscapes_label_colormap(),
    "mapillary": create_mapillary_vistas_label_colormap(),
    "ade20k": create_ade20k_label_colormap(),
    "voc": create_pascal_label_colormap(),
    "coco": create_coco_stuff_colormap(),
}
# json 으로부터 shot이 변경되는 부분의 frame을 읽어오기
shot_changed = []
with open('MiSang_Frame.json') as json_file:
	misang_frame = json.load(json_file)

for i in range(len(misang_frame)):
    shot_changed.append(misang_frame[i]['frame']-1)
shot_changed.remove(0) # 첫번째 쓰레기값 제거, shot이 변경되는 순간의 frame
print(shot_changed)
# ----------------------------------------------------------------------------------------------------------------------
# rule1 : 이건 디텍트론이 객체를 찾아주는 거라서 yolo의 결과와는 다소 차이가 있음 yolo의 결과가 더 좋음
tmp = [0] * 80  # 80개의 cocodataset을 담을 변수 만들기
frame = [1]     # 몇번째 frmae인지를 표시하기 위한 변수
shot = {}       # shot의 입력에 따라 순서대로 정보를 저장할 dictionary 변수
shot_count = [1]
# shot_changed = [3,6,10,15,20]          # shot이 변경될 때의 frame을 입력해주면 됨!, 추후 json으로!
# 80개 cocodataset을 reset하는 code
def reset(tmp):
    for i in range(len(tmp)):
        tmp[i] = 0
def calc(shot):
    ct1 = 0
    ct2 = 0
    for i in range(len(shot['shot_1'])):
        if (shot['shot_' + str(len(shot.keys())-1)][i] == 1):
            ct1 += 1
        if(shot['shot_' + str(len(shot.keys()))][i] == 1):
            ct2 += 1
    max_value = max(ct1,ct2)
    return max_value
# AND연산 수행
def compare(shot):
    # 샷의 갯수 중 끝에서 2개끼리 비교하도록 하면 됨...
    count = 0
    for i in range(len(shot['shot_1'])):
        if ((shot['shot_' + str(len(shot.keys())-1)][i] & shot['shot_' + str(len(shot.keys()))][i]) == 1):
            count += 1
    # print('같은 클래스의 수 : ', count)
    max_value = calc(shot)  # 큰 값음 함수로 계산
    similarity1 = count / max_value
    print('Similarity1         : ', similarity1)
    return  similarity1

def rule1(classes,class_names,temp1):
    print("---------------------------------------------------Descriprion_Rule1-----------------------------------------------------")
    # print("classes     : ",classes)               # 한 frame에서 검출한 객체의 index를 모두 담고있음
    result = set(classes)                         # 중복되는 class를 정리
    print('set_classes : ', result)               # 정리한 클래스 출력
    # print('index_match : ',tmp)                   # 80개의 list에 검출한 index를 누적해서 표시!
    # print('frame_number: ', len(frame))
    temp = tmp.copy()
    # shot 별로 매칭된 80개의 리스트를 누적해 딕셔너리에 저장하는 코드
    for k in range(len(shot_changed)):
        #if(len(frame) == shot_changed[len(shot_count)-1]):
        if(len(frame) == shot_changed[k]):
            #shot = {'shot_'+str(len(shot_count)) : tmp}
            shot['shot_' + str(len(shot_count))] = temp # 샷이 바뀔 때마다 딕셔너리에 80개의 class list를 저장
            shot_count.append(1)
            print(shot) # 누적되는 샷을 출력
            reset(tmp)
    # print('shot_count : ', len(shot.keys()))            # 몇번째 샷인지 출력, 여기까지 문제 없음

    # and 연산 수행
    if (len(shot.keys()) >= 2):
        similarity1 = compare(shot)
        return similarity1
# ----------------------------------------------------------------------------------------------------------------------
# rule2_initialization
tmp1 = [0] * 80  # 80개의 cocodataset을 담을 변수 만들기
temp_sim2 = [0] * 80 # 80개의 각 class에 대한 차이를 저장할 변수(분자)
temp_sim3 = [0] * 80 # 80개의 각 class에 대해 큰 수를 자장할 변수(분모)
sim_2 = [0] * 80 # 80개의 각 class에 대한 유사도
sim_total = 0
dict = {}
shot_count1 = [1]
def calc2(dict):
    for i in range(len(dict['shot_1'])):
        temp_sim3[i] = max(dict['shot_' + str(len(dict.keys())-1)][i],dict['shot_' + str(len(dict.keys()))][i])
        if((temp_sim2[i]!=0) & (temp_sim3[i] != 0)):
            sim_2[i] = 1 - (temp_sim2[i]/temp_sim3[i])
    print("큰 값( 두 class) : ", temp_sim3)
    print("80개 유사도", sim_2)
    count = 0 # 존재하는 유사도가 몇개인지?
    for j in range(len(sim_2)):
        if(sim_2[j] != 0):
            count += 1
    if(count == 0):
        similarity2 = 0
    else:
        similarity2 = sum(sim_2)/count              # 이미 1에서 뺀 값이므로 평균만 내어주면 됨
    print("similarity2 : ", similarity2)
    return similarity2

def compare2(dict):
    for i in range(len(dict['shot_1'])):
        temp_sim2[i] = abs(dict['shot_' + str(len(dict.keys()) - 1)][i] - dict['shot_' + str(len(dict.keys()))][i])
    print("이전샷 - 현재샷(분자) : ", temp_sim2)
    simil_2 = calc2(dict)
    return simil_2

# rule2
def rule2(temp1):
    print("---------------------------------------------------Descriprion_Rule2-----------------------------------------------------")
    print("temp1 : ", temp1)                    # 한프레임에서 검출한 모든 객체의 index_number
    # 80개 리스트에 갯수를 누적 저장
    for i in range(len(temp1)):
        tmp1[temp1[i]] += 1
    print("80개 리스트 : ",tmp1)
    temp2 = tmp1.copy()                         # 저장을 위해 복사
    # 샷이 바뀔 때마다 dict에 class별 index를 누적한 list를 저장
    for k in range(len(shot_changed)):
        if(len(frame) == shot_changed[k]):
            dict['shot_' + str(len(shot_count1))] = temp2 # 샷이 바뀔 때마다 딕셔너리에 80개의 class list를 저장
            shot_count1.append(1)
            reset(tmp1)
    print("rule2 : ", dict)
    # 샷이 2개 이상이 되면 유사도 도출을 시작
    if (len(dict.keys()) >= 2):
        similarity2 = compare2(dict)
        return similarity2
# ----------------------------------------------------------------------------------------------------------------------
# Standardization(도출한 유사도를 활용하는 부분!! 일단 전처리는 안하고 평균값을 사용!)
simil1 = []
simil2 = []
similarity_average = []
similar1 = {}
similar1['similarity1'] = []
similar2 = {}
similar2['similarity2'] = []
similarity_av = {}
similarity_av['similarity_average'] = []
similarity2 = {}
def standardization(similarity1, similarity2, frame):
    # 유사도가 없는 경우(shot이 최소 2개가 되지 않을 경우) -> 유사도를 0
    if(similarity1 == None):
        similarity1 = 0
    if(similarity2 == None):
        similarity2 = 0
    print("유사도1 : ", similarity1)
    print("유사도2 : ", similarity2)
    print(len(frame))
    simil_average = (similarity1 + similarity2) / 2
    for k in range(len(shot_changed)):
        if(len(frame) == shot_changed[k]):
            similarity_average.append(simil_average)        # 첫번때 인덱스에 저장되는 값은 버려야 함(쓰레기 값임)
            simil1.append(similarity1)
            simil2.append(similarity2)
    print("평균 유사도 : ", simil_average)
    print("similarity_total", similarity_average)
    print("similarity1 : ", simil1)
    print("similarity2 : ", simil2)
    # 유사도1,2 저장(첫번째 inidex 날려야함)
    for i in range(len(shot_changed)):
        # if(len(frame) == shot_changed[len(shot_count)-1]):
        if (len(frame) == shot_changed[i]):
            similar1['similarity1'].append(similarity1)
            similar2['similarity2'].append(similarity2)
            similarity_av['similarity_average'].append(simil_average)
            with open('similarity1.json', 'w', encoding="utf-8") as outfile:
                json.dump(similar1, outfile)
            with open('similarity2.json', 'w', encoding="utf-8") as outfile:
                json.dump(similar2, outfile)
            with open('similarity_average.json', 'w', encoding="utf-8") as outfile:
                json.dump(similarity_av, outfile)
    # 여기서 장면분할을 위한 함수를 만들어 매개변수 던지기


    # del simil1[0]
    # del simil2[0]
    # print(simil1)       # 유사도를 누적한 리스트의 0번째 값은 버려야함
    # print(simil2)       # 유사도를 누적한 리스트의 0번째 값은 버려야함
    # 1.단순 평균으로 종합 유사도 도출


    # 2.전처리 공식에 넣어서 유사도 도출 -> 유사도의 값이 모두 0~1로 맞춰지기 때문에 평균으로 일단 test
    # scaler = StandardScaler()
    # simil1_scaled = scaler.transform(simil1)
    return similarity_average


def draw_masks_maskrcnn(image, boxes, scores, labels, masks, human_label_list=None,
                        score_thresh=0.6, draw_box=True):
    """
    Standared mask drawing function

    boxes: a list of boxes, or numpy array
    scores: a list of scores or numpy array
    labels: same as scores
    masks: resize to same width and height as box masks

    NOTE: if masks not same with box, then it will resize inside this function

    this function has speed issue be kind of slow.
    and it overlays a box on another mask rather than mask by mask, this is not best
    way.

    :param image:
    :param boxes:
    :param scores:
    :param labels:
    :param masks:
    :param human_label_list
    :param score_thresh
    :param draw_box:
    :return:
    """
    n_instances = 0
    if isinstance(boxes, list):
        n_instances = len(boxes)
    else:
        n_instances = boxes.shape[0]
    # black image with same size as original image
    empty_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(n_instances):
        box = boxes[i]
        score = scores[i]
        label = labels[i]
        mask = masks[i]

        cls_color = get_unique_color_by_id(label)
        # only get RGB
        instance_color = get_unique_color_by_id(i)[:-1]
        # now adding masks to image, and colorize it
        if score >= score_thresh:

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            if draw_box:
                image = draw_one_bbox(image, box, cls_color, 1)
                if human_label_list:
                    # draw text on image
                    font = cv2.QT_FONT_NORMAL
                    font_scale = 0.4
                    font_thickness = 1
                    line_thickness = 1

                    txt = '{} {:.2f}'.format(human_label_list[label], score)
                    cv2.putText(image, txt, (x1, y1), font,
                                font_scale, cls_color, font_thickness)

            # colorize mask
            m_w = int(x2-x1)
            m_h = int(y2-y1)
            mask = Image.fromarray(mask).resize((m_w, m_h), Image.LINEAR)
            mask = np.array(mask)
            # cv2.imshow('rr2', mask)
            # cv2.waitKey(0)

            mask_flatten = mask.flatten()
            # if pixel value less than 0.5, that's background, min: 0.0009, max: 0.9
            mask_flatten_color = np.array(list(map(lambda it: instance_color if it > 0.5 else [0, 0, 0],
                                                   mask_flatten)), dtype=np.uint8)

            mask_color = np.resize(mask_flatten_color, (m_h, m_w, 3))
            empty_image[y1: y2, x1: x2, :] = mask_color
    # combine image and masks
    # now we got mask
    combined = cv2.addWeighted(image, 0.5, empty_image, 0.6, 0)
    return combined


def draw_masks_maskrcnn_v2(image, boxes, scores, labels, masks, human_label_list=None,
                           score_thresh=0.6, draw_box=True):
    """
    We change way to draw masks on image

    :param image:
    :param boxes:
    :param scores:
    :param labels:
    :param masks:
    :param human_label_list
    :param score_thresh
    :param draw_box:
    :return:
    """
    n_instances = 0
    if isinstance(boxes, list):
        n_instances = len(boxes)
    else:
        n_instances = boxes.shape[0]
    # black image with same size as original image
    empty_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(n_instances):
        box = boxes[i]
        score = scores[i]
        label = labels[i]
        mask = masks[i]

        cls_color = get_unique_color_by_id(label)
        # only get RGB
        instance_color = get_unique_color_by_id(i)[:-1]
        # now adding masks to image, and colorize it
        if score >= score_thresh:

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            if draw_box:
                image = draw_one_bbox(image, box, cls_color, 1)
                if human_label_list:
                    # draw text on image
                    font = cv2.QT_FONT_NORMAL
                    font_scale = 0.4
                    font_thickness = 1
                    line_thickness = 1

                    txt = '{} {:.2f}'.format(human_label_list[label], score)
                    cv2.putText(image, txt, (x1, y1), font,
                                font_scale, cls_color, font_thickness)

            # colorize mask
            m_w = int(x2-x1)
            m_h = int(y2-y1)
            mask = Image.fromarray(mask).resize((m_w, m_h), Image.LINEAR)
            mask = np.array(mask)
            mask_flatten = mask.flatten()
            # if pixel value less than 0.5, that's background, min: 0.0009, max: 0.9
            mask_flatten_color = np.array(list(map(lambda it: instance_color if it > 0.5 else [0, 0, 0],
                                                   mask_flatten)), dtype=np.uint8)

            mask_color = np.resize(mask_flatten_color, (m_h, m_w, 3))
            empty_image[y1: y2, x1: x2, :] = mask_color
    # combine image and masks
    # now we got mask
    combined = cv2.addWeighted(image, 0.5, empty_image, 0.6, 0)
    return combined


# more fast mask drawing here
def vis_bitmasks(img, bitmasks, classes=None, fill_mask=True, return_combined=True, thickness=1, draw_contours=True):
    """
    visualize bitmasks on image
    """
    # need check if img and bitmask with same W,H
    if isinstance(bitmasks, torch.Tensor):
        bitmasks = bitmasks.cpu().numpy()

    font = cv2.QT_FONT_NORMAL
    font_scale = 0.4
    font_thickness = 1
    res_m = np.zeros_like(img).astype(np.uint8)
    assert isinstance(bitmasks, np.ndarray), 'bitmasks must be numpy array'
    bitmasks = bitmasks.astype(np.uint8)
    for i, m in enumerate(bitmasks):
        if m.shape != img.shape:
            m = cv2.resize(m, (img.shape[1], img.shape[0]))
        cts, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # inssue this is a unique color
        if classes is not None:
            c = get_unique_color_by_id2(classes[i])
        else:
            c = get_unique_color_by_id2(i)
        if return_combined:
            if fill_mask:
                cv2.drawContours(res_m, cts, -1,  color=c,
                                 thickness=-1, lineType=cv2.LINE_AA)
                if draw_contours:
                    cv2.drawContours(img, cts, -1,  color=c,
                                     thickness=thickness, lineType=cv2.LINE_AA)
            else:
                cv2.drawContours(res_m, cts, -1,  color=c,
                                 thickness=thickness, lineType=cv2.LINE_AA)
        else:
            if fill_mask:
                cv2.drawContours(img, cts, -1,  color=c,
                                 thickness=-1, lineType=cv2.LINE_AA)
            else:
                cv2.drawContours(img, cts, -1,  color=c,
                                 thickness=thickness, lineType=cv2.LINE_AA)
    if return_combined:
        img = cv2.addWeighted(img, 0.6, res_m, 0.7, 0.8)
        return img
    else:
        return img


def vis_bitmasks_with_classes(img, classes, bitmasks, force_colors=None, scores=None, class_names=None, mask_border_color=None, draw_contours=False, alpha=0.4, fill_mask=True, return_combined=True, thickness=2):
    """
    visualize bitmasks on image
    """
    #initialization
    temp1 = [] # class_names를 누적할 변수
    # need check if img and bitmask with same W,H
    if isinstance(bitmasks, torch.Tensor):
        bitmasks = bitmasks.cpu().numpy()

    if class_names is None or len(class_names) == 0:
        class_names = coco_label_map_list[1:]
    #print(coco_label_map_list[1]) # coco index는 1번부터 80까지 총 80개임
    #print(coco_label_map_list[0]) # 0번째 index는 background임

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    res_m = np.zeros_like(img)
    assert isinstance(bitmasks, np.ndarray), 'bitmasks must be numpy array'
    bitmasks = bitmasks.astype(np.uint8)
    for i, m in enumerate(bitmasks):
        if m.shape != img.shape:
            m = cv2.resize(m, (img.shape[1], img.shape[0]))
        cts, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(cts) > 0:
            cts = max(cts, key=cv2.contourArea)
        # enssue this is a unique color
        cid = int(classes[i])
        temp1.append(cid)
        #print("cid : ", cid) # 찾은 class의 index를 저장하고 있음(프레임별로 갯수별로 담고있음
        # print("class_names", class_names) # cocodataset의 80개 class 목록
        # print("index_length", len(class_names))
        if force_colors:
            c = force_colors[cid]
        else:
            c = get_unique_color_by_id2(cid)
        if len(cts) > 0:
            if return_combined:
                if fill_mask:
                    cv2.drawContours(res_m, [cts], -1,  color=c,
                                    thickness=-1, lineType=cv2.LINE_AA)
                    if draw_contours:
                        if mask_border_color:
                            c = mask_border_color
                        cv2.drawContours(img, [cts], -1,  color=c,
                                        thickness=thickness, lineType=cv2.LINE_AA)
                else:
                    cv2.drawContours(res_m, [cts], -1,  color=c,
                                    thickness=thickness, lineType=cv2.LINE_AA)
            else:
                if fill_mask:
                    cv2.drawContours(img, [cts], -1,  color=c,
                                    thickness=-1, lineType=cv2.LINE_AA)
                else:
                    cv2.drawContours(img, [cts], -1,  color=c,
                                    thickness=thickness, lineType=cv2.LINE_AA)
        if classes is not None:
            txt = f'{class_names[classes[i]]}'
            if scores is not None:
                txt += f' {scores[classes[i]]}'
            if len(cts) > 0:
                M = cv2.moments(cts)
                cx = None
                cy = None
                if (M["m00"] == 0):
                    cx = 0
                    cy = 0
                else:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                # draw labels
                # print(class_names[classes[i]]) # 전체 class 이름
                # print(class_names[0])
                # print(classes[i]) #검출한 class의 index number
                tmp[classes[i]] = 1
                cv2.putText(img, txt, (cx, cy), font, font_scale, [255, 255, 255], 1, cv2.LINE_AA)
    # 유사도 도출
    similarity1 = rule1(classes, class_names, temp1)  # rule 1번
    similarity2 = rule2(temp1)
    # print("유사도1 : ", similarity1) #현재 값
    # print("유사도2 : ", similarity2) #현재 값

    # 표준화 전처리(Standardization)
    similarity_total= standardization(similarity1,similarity2,frame)
    # print("평균 유사도 : ", similarity_total)

    # 장면 분할(scene_segmentation)

    frame.append(1)
    if return_combined:
        img = cv2.addWeighted(img, 0.7, res_m, alpha, 0.4)
        return img
    else:
        return img


# helper functions
def label2color_mask(cls_id_mask, max_classes=90, override_id_clr_map=None, color_suit='cityscapes'):
    """
    cls_id_mask is your segmentation output
    override_id_clr_map: {2: [0, 0, 0]} used to override color
    """
    assert color_suit in ALL_COLORS_MAP.keys(
    ), 'avaiable keys: {}'.format(ALL_COLORS_MAP.keys())
    colors = ALL_COLORS_MAP[color_suit]
    if override_id_clr_map != None:
        if isinstance(override_id_clr_map, dict):
            colors = np.array([get_unique_color_by_id_with_dataset(i) if i not in override_id_clr_map.keys(
            ) else override_id_clr_map[i] for i in range(max_classes)])
        else:
            colors = np.array([override_id_clr_map[i % len(
                override_id_clr_map)] for i in range(max_classes)])
    if len(colors) < max_classes:
        colors = np.append(
            colors, ALL_COLORS_MAP['cityscapes'][:max_classes-len(colors)], axis=0)

    s = cls_id_mask.shape
    if len(s) > 1:
        cls_id_mask = cls_id_mask.flatten()
    mask = colors[cls_id_mask]
    mask = np.reshape(mask, (*s, 3)).astype(np.uint8)
    return mask