# generate the json file for multiple training
# json file for image path
# json file of hand postion output for three different model
# evaluate hand bbox result and return reliable data

import json
import os
from tqdm import tqdm

def genFromPath(path="../data/vlog/all/images", imgFormat='jpg'):
    g = os.walk(path)
    imgLst = []
    for path, dirLst, fileLst in g:
        for fileName in tqdm(fileLst):
            imgLst.append(os.path.join('vlog/all/images', fileName))
    # for local test
    imgLst = imgLst[0:10]
    json.dump(imgLst, open('../data/vlog_imgs.json', 'w'))

def genFromFile(path="../hand_state_final.txt", imgFormat='jpg'):
    imgLst = []
    usageLst = []
    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            line = line.split(' ')
            imgLst.append(line[0])
            usageLst.append(line[1])
    json.dump([imgLst, usageLst], open('../data/complete_vlog_imgs.json', 'w'))

def genHMRWrist(writs):
    # writs = {'<fileName>': {'left_wrist': [x, y], 'right_wrist': [x, y]}...}
    with open('../data/wrists.json', 'w') as f:
        json.dump(writs, f)

def genDetecBboxes(bboxes):
    # bboxes = {'<fileName>': {bboxes': [[x0, y0, x1, y1]...], 'score': [...]}...}
    with open('../data/detect_bboxes.json', 'w') as f:
        json.dump(bboxes, f)

def genObjBboxes(bboxes):
    # bboxes = {'<fileName>': {'filename': <fileName>, 'bboxes': [[x0, y0, x1, y1]...], 'lr': [<isRight>...], 'score': [...]}...}
    with open('../data/handobj_bboxes.json', 'w') as f:
        json.dump(bboxes, f)

if __name__ == '__main__':
    genFromPath()
    genFromFile()

