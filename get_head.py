import json
import cv2

def cat(img, x, y, w, h):
    height, width = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x+w), min(height, y+h)
    return img[y0:y1, x0:x1]

def get_xy(box):
    return int(min(box[0][0], box[3][0])), int(min(box[0][1], box[1][1]))

def parse_msg(msg):
    name_box, addr_box = None, None
    for value in msg:
        box, text = value[0], value[1][0]
        if "姓" in text and "名" in text:
            name_box = box
            continue
        if "住" in text and "址" in text:
            addr_box = box
            continue
    return name_box, addr_box
        

def get_avatar(img, msg):
    name_box, addr_box = parse_msg(msg)
    print(name_box, addr_box)
    if name_box is None or addr_box is None:
        return
    x0, y0 = get_xy(name_box)
    x1, y1 = get_xy(addr_box)
    base = y1 - y0
    x, y = x0 + int(2.4 * base), y0
    w, h = int(1.25 * base), int(1.66 * base)
    return cat(img, x, y, w, h)


with open("result.json") as f:
    result = json.load(f)
    print(result)

img = cv2.imread("test_img/z1.jpg")

cv2.imshow("", get_avatar(img, result))
cv2.waitKey()