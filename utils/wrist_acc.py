import numpy as np
import json
from tqdm import tqdm
import cv2
import copy


def bbox_test(wrist, bbox, scale=1.0):
    dx = bbox[2] - bbox[0]
    dy = bbox[3] - bbox[1]
    dx_scaled = (scale - 1.0) * dx / 2.0
    dy_scaled = (scale - 1.0) * dy / 2.0
    bbox_scaled = [bbox[0] - dx_scaled, bbox[1] -
                   dy_scaled, bbox[2] + dx_scaled, bbox[3] + dy_scaled]
    return (wrist[0] > bbox_scaled[0] and wrist[0] < bbox_scaled[2]) and (wrist[1] > bbox_scaled[1] and wrist[1] < bbox_scaled[3])


def d2_acc(hmr_json, d2_json):
    hmr_data = open(hmr_json)
    hmr_wrists = json.load(hmr_data)
    d2_data = open(d2_json)
    d2_bboxes = json.load(d2_data)

    total_num = len(hmr_wrists)
    l_correct_num = 0
    r_correct_num = 0
    not_found_img = 0
    for img in tqdm(hmr_wrists.keys()):
        if d2_bboxes[img] is None:
            print("image {0} does not exist in detector result!".format(img))
            not_found_img += 1
            continue

        bboxes = d2_bboxes[img]['bboxes']
        l_wrist = hmr_wrists[img]['left_wrist']
        r_wrist = hmr_wrists[img]['right_wrist']

        l_count = 0
        r_count = 0

        for bbox in bboxes:
            if bbox_test(l_wrist, bbox, 1.2):
                l_count += 1
            if bbox_test(r_wrist, bbox, 1.2):
                r_count += 1

        # print('{0} : {1}'.format(l_count, r_count))
        if (l_count > 0):
            l_correct_num += 1
        if (r_count > 0):
            r_correct_num += 1

    l_acc = float(l_correct_num) / float(total_num - not_found_img)
    r_acc = float(r_correct_num) / float(total_num - not_found_img)
    return l_acc, r_acc


def obj_acc(hmr_json, obj_json):
    hmr_data = open(hmr_json)
    hmr_wrists = json.load(hmr_data)
    obj_data = open(obj_json)
    obj_bboxes = json.load(obj_data)

    total_num = len(hmr_wrists)
    l_correct_num = 0
    r_correct_num = 0
    not_found_img = 0
    for img in tqdm(hmr_wrists.keys()):
        if obj_bboxes[img] is None:
            print("image {0} does not exist in detector result!".format(img))
            not_found_img += 1
            continue

        bboxes = obj_bboxes[img]['bboxes']
        lr_info = obj_bboxes[img]['lr']
        l_wrist = hmr_wrists[img]['left_wrist']
        r_wrist = hmr_wrists[img]['right_wrist']

        l_count = 0
        r_count = 0
        if len(lr_info) is not len(bboxes):
            print('image {0} size does not match'.format(img))

        for i in range(len(lr_info)):
            bbox = bboxes[i]
            lr = lr_info[i]
            if not lr and bbox_test(l_wrist, bbox, 1.2):
                l_count += 1
            elif bbox_test(r_wrist, bbox, 1.2):
                r_count += 1

        if (l_count > 0):
            l_correct_num += 1
        if (r_count > 0):
            r_correct_num += 1

    l_acc = float(l_correct_num) / float(total_num - not_found_img)
    r_acc = float(r_correct_num) / float(total_num - not_found_img)
    return l_acc, r_acc


def img_shift_padding(img, shift=[-20, -10, 0, 10, 20]):
    shape = img.shape
    assert(len(shape) == 3 and shape[2] == 3)
    res = []
    for s in shift:
        img_cpy = copy.deepcopy(img)
        img_cpy = np.roll(np.roll(img_cpy, s, axis=0), s, axis=1)
        r = np.zeros(shape)
        if s >= 0:
            r[s:, s:] = img_cpy[s:, s:]
        else:
            r[:s, :s] = img_cpy[:s, :s]
        res.append(r.astype(np.uint8))

    return res


if __name__ == '__main__':
    l_acc, r_acc = d2_acc('../data/wrists.json', '../data/detect_bboxes.json')
    print('left accuracy: {0}, right accuracy: {1}'.format(l_acc, r_acc))

    l_acc, r_acc = obj_acc('../data/wrists.json',
                           '../data/handobj_bboxes.json')
    print('left accuracy: {0}, right accuracy: {1}'.format(l_acc, r_acc))

    # img shift padding test
    img = cv2.imread('../hmr/demo/vlog_q_u_Q_v_qNSfZz0HquQ_017_frame000151.jpg')

    shift=[-20, -10, 0, 10, 20]
    res = img_shift_padding(img, shift)

    for i in range(5):
        cv2.imwrite('../data/shifted_' + str(shift[i]) + '.png', res[i])
