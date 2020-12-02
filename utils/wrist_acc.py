import numpy as np
import json
from tqdm import tqdm


def bbox_test(wrist, bbox, scale = 1.0):
    bbox_scaled = [coord * scale for coord in bbox]
    return (wrist[0] > bbox_scaled[0] and wrist[0] < bbox_scaled[2]) and (wrist[1] > bbox_scaled[1] and wrist[1] < bbox_scaled[3])

def d2_acc(hmr_json, d2_json):
    hmr_data = open(hmr_json)
    hmr_wrists = json.load(hmr_data)
    d2_data = open(d2_json)
    d2_bboxes = json.load(d2_data)

    total_num = len(hmr_wrists)
    correct_num = 0
    not_found_img = 0
    for img in tqdm(hmr_wrists.keys()):
        if d2_bboxes[img] is None:
            print("image {img} does not exist in detector result!")
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
        if (l_count > 0 and r_count > 0):
            correct_num += 1

    acc = float(correct_num) / float(total_num - not_found_img)
    return acc


if __name__ == '__main__':
    acc = d2_acc('../data/wrists.json', '../data/detect_bboxes.json')
    print(acc)

