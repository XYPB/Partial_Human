import numpy as np
import json
from tqdm import tqdm


def bbox_test(wrist, bbox):
    return (wrist[0] > bbox[0] and wrist[0] < bbox[2]) and (wrist[1] > bbox[1] and wrist[1] < bbox[3])

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

        bboxes = d2_bboxes[img]
        l_wrist = hmr_wrists[img]['left_wrist']
        r_wrist = hmr_wrists[img]['right_wrist']

        l_count = 0
        r_count = 0

        for bbox in bboxes:
            if bbox_test(l_wrist, bbox):
                l_count += 1
            if bbox_test(r_wrist, bbox):
                r_count += 1

        if (l_count == 1 and r_count == 1) or (l_count == 2 and r_count == 2):
            correct_num += 1

    acc = float(correct_num) / float(total_num - not_found_img)
    return acc


if __name__ == '__main__':
    acc = d2_acc('../data/wrists.json', '../data/detect_bboxes.json')
    print(acc)

