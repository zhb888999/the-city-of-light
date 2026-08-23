import numpy as np
import onnxruntime
import cv2
from shapely.geometry import Polygon
import pyclipper
import re
import copy
import math
import base64
import logging
import asyncio
from typing import List
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse


class ONNXRunner:

    def __init__(self, model_path, inputs=None, outputs=None) -> None:
        options = onnxruntime.SessionOptions()
        options.enable_profiling=True
        self.sess = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # self.sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.inputs = [input.name for input in self.sess.get_inputs()] if inputs is None else inputs
        self.outputs = [output.name for output in self.sess.get_outputs()] if outputs is None else outputs

    def __call__(self, inputs):
        return self.sess.run( self.outputs, {
            input_name: input for input_name, input in zip(self.inputs, inputs)
        })

class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(
        self,
        max_candidates=1000,
        unclip_ratio=2.0,
        use_dilation=False,
        **kwargs,
    ):
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])


    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height, box_thresh=0.7):
        """
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        """

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score(pred, points.reshape(-1, 2))
            if box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio)
            if len(box) > 1:
                continue
            box = np.array(box).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score(self, bitmap, _box):
        """
        box_score_fast: use bbox mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def __call__(self, pred, shape_list, thresh=0.3, box_thresh=0.7):
        pred = pred[:, 0, :, :]
        segmentation = pred > thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel,
                )
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(
                pred[batch_index], mask, src_w, src_h, box_thresh=box_thresh
            )

            boxes_batch.append({"points": boxes})
        return boxes_batch




class BaseRecLabelDecode(object):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if "arabic" in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)

        return "".join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def get_word_info(self, text, selection):
        """
        Group the decoded characters and record the corresponding decoded positions.

        Args:
            text: the decoded text
            selection: the bool array that identifies which columns of features are decoded as non-separated characters
        Returns:
            word_list: list of the grouped words
            word_col_list: list of decoding positions corresponding to each character in the grouped word
            state_list: list of marker to identify the type of grouping words, including two types of grouping words:
                        - 'cn': continous chinese characters (e.g., 你好啊)
                        - 'en&num': continous english characters (e.g., hello), number (e.g., 123, 1.123), or mixed of them connected by '-' (e.g., VGG-16)
                        The remaining characters in text are treated as separators between groups (e.g., space, '(', ')', etc.).
        """
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection == True)[0]

        for c_i, char in enumerate(text):
            if "\u4e00" <= char <= "\u9fff":
                c_state = "cn"
            elif bool(re.search("[a-zA-Z0-9]", char)):
                c_state = "en&num"
            else:
                c_state = "splitter"

            if (
                char == "."
                and state == "en&num"
                and c_i + 1 < len(text)
                and bool(re.search("[0-9]", text[c_i + 1]))
            ):  # grouping floting number
                c_state = "en&num"
            if (
                char == "-" and state == "en&num"
            ):  # grouping word with '-', such as 'state-of-the-art'
                c_state = "en&num"

            if state == None:
                state = c_state

            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            if state != "splitter":
                word_content.append(char)
                word_col_content.append(valid_col[c_i])

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return word_list, word_col_list, state_list

    def decode(
        self,
        text_index,
        text_prob=None,
        is_remove_duplicate=False,
        return_word_box=False,
    ):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(
                    text, selection
                )
                result_list.append(
                    (
                        text,
                        np.mean(conf_list).tolist(),
                        [
                            len(text_index[batch_idx]),
                            word_list,
                            word_col_list,
                            state_list,
                        ],
                    )
                )
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char)

    def filter(self, letters, pos):
        arr = np.array(pos)
        diff = np.diff(arr)
        ignore_index = []
        for i, coincidental in enumerate(diff < np.floor(np.average(diff))):
            confuse_letters = ["0", "O", "o"]
            if coincidental:
                if letters[i] in confuse_letters and letters[i+1] in confuse_letters:
                    ignore_index.append(i+1)
        new_letters, new_pos = [], []
        for i in range(len(letters)):
            if i in ignore_index:
                continue
            new_letters.append(letters[i])
            new_pos.append(pos[i])
        return new_letters, new_pos

    def filter_texts(self, texts, return_word_box=False):
        new_texts = []
        for text in texts:
            if text[0] == "":
                new_texts.append(text)
                continue
            if len(text[2][1]) == 0 or len(text[2][2]) == 0:
                new_texts.append(text)
                continue
            # letters, pos = self.filter(text[2][1][0], text[2][2][0])
            letters, pos = text[2][1][0], text[2][2][0]
            string = "".join(letters)
            new_text = ("".join(letters), text[1], [text[2][0], [letters], [pos]]) if return_word_box else ("".join(letters), text[1])
            new_texts.append(new_text)
        return new_texts

    def __call__(self, preds, return_word_box=False, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
            return_word_box=True,
        )

        text = self.filter_texts(text, return_word_box=return_word_box)
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs["wh_ratio_list"][rec_idx]
                max_wh_ratio = kwargs["max_wh_ratio"]
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)
        return text

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character



class DetResizeForTest(object):
    def __init__(self, limit_side_len=960):
        self.limit_side_len = limit_side_len

    def __call__(self, img):
        src_h, src_w, _ = img.shape
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)
        img, [ratio_h, ratio_w] = self.resize_image(img)
        return img, np.array([src_h, src_w, ratio_h, ratio_w])

    def image_padding(self, im, value=0):
        h, w, c = im.shape
        im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
        im_pad[:h, :w, :] = im
        return im_pad

    def resize_image(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, c = img.shape

        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            ratio = 1.0

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]



class NormalizeAndTransposeImage:

    def __init__(self):
        self.scale = np.float32(1.0 / 255.0)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.mean = np.array(mean).reshape((1, 1, 3)).astype("float32")
        self.std = np.array(std).reshape((1, 1, 3)).astype("float32")

    def __call__(self, image):
        image =  (image.astype(np.float32) * self.scale - self.mean) / self.std
        return image.transpose((2, 0, 1))

class Detector:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model_runner = ONNXRunner(self.model_path)
        self.preprocess_transform0 = DetResizeForTest()
        self.preprocess_transform1= NormalizeAndTransposeImage()
        self.postprocess = DBPostProcess(use_dilation=True)

    def preprocess(self, image):
        image, shape_list = self.preprocess_transform0(image)
        return self.preprocess_transform1(image), shape_list

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, image, threshold=0.3, box_threshold=0.7):
        image_shape = image.shape
        image, shape_list = self.preprocess(image)
        image = np.expand_dims(image, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        outputs = self.model_runner([image])
        output = 1 / (1 + np.exp(-outputs[0]))
        post_result = self.postprocess(output, shape_list, threshold, box_threshold)
        dt_boxes = post_result[0]["points"]
        return self.filter_tag_det_res(dt_boxes, image_shape)


class Recognizer:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model_runner = ONNXRunner(model_path)
        self.postprocess = CTCLabelDecode("keys_v1.txt", True)
        self.rec_batch_num = 16
        self.rec_image_shape = [3, 48, 320]
        self.return_word_box = False

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(
                    img_list[indices[ino]], max_wh_ratio
                )
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            preds = self.model_runner([norm_img_batch])
            rec_result = self.postprocess(
                preds,
                return_word_box=self.return_word_box,
                wh_ratio_list=wh_ratio_list,
                max_wh_ratio=max_wh_ratio,
            )
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res

class OCRModel:

    def __init__(self, detector_model, recognizer_model):
        self.detector = Detector("model/detsub.onnx")
        self.recongnizer = Recognizer("model/rec.onnx")

    @staticmethod
    def sorted_boxes(dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes


    @staticmethod
    def get_rotate_crop_image(img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def __call__(self, image, threshold=0.3, box_threshold=0.7):
        boxes = self.detector(image, threshold, box_threshold)
        boxes = self.sorted_boxes(boxes)
        img_crops = []
        for box in boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = self.get_rotate_crop_image(image, tmp_box)
            # cv2.imshow("", img_crop)
            # cv2.waitKey()
            img_crops.append(img_crop)
        rec_res = self.recongnizer(img_crops)
        return [(box.tolist(), res) for box, res in zip(boxes, rec_res)]


def test():
    import json
    ocr = OCRModel("model/detsub.onnx", "model/rec.onnx")
    # img = cv2.imread("test_img/4.png")
    img = cv2.imread("test_img/4.jpg")
    result = ocr(img, 0.3, 0.6)
    print(result)
    # with open("result.json", "w") as f:
    #     json.dump(result, f)


def base642array(b64):
    image = base64.b64decode(b64)
    image = np.frombuffer(image, np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

class Msg(BaseModel):
    image: str
    threshold: float = 0.2
    box_threshold: float = 0.7

ocr = OCRModel("model/detsub.onnx", "model/rec.onnx")
app = FastAPI(docs_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

config = {
    "only_treasure_chest": False,
    "jump_paint": False,
    "jump_nail_board": False,
    "close_service": False,
    "refresh_time": 1.0,
    "adb_ip": "192.168.3.206",
    "adb_port": 5555,
    "network_adb": True
}
lock = asyncio.Lock()

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=config.copy()
    )

@app.get("/config")
async def index(request: Request):
    return config.copy()

@app.post("/config")
async def set_config(
    refresh_time: float = Form(...), 
    adb_ip: str = Form(...),
    adb_port: int = Form(int),
    network_adb: List[str] = Form(default=[]),
    only_treasure_chest: List[str] = Form(default=[]),
    jump_nail_board: List[str] = Form(default=[]),
    jump_paint: List[str] = Form(default=[]),
    close_service: List[str] = Form(default=[])):
    global config
    async with lock:
        config["adb_ip"] = adb_ip
        config["adb_port"] = adb_port
        config["network_adb"] = len(network_adb) > 0
        config["only_treasure_chest"] = len(only_treasure_chest) > 0
        config["jump_paint"] = len(jump_paint) > 0
        config["jump_nail_board"] = len(jump_nail_board) > 0
        config["close_service"] = len(close_service) > 0
    return RedirectResponse(url="/", status_code=303) 

@app.post('/model/ocr')
async def ocr_server(msg: Msg):
    try:
        image = base642array(msg.image)
        result = ocr(image, msg.threshold, msg.box_threshold)
    except Exception as e:
        logging.error(f"runtime error, exception={e}")
        return {"code": 1, "msg": "unknow error", "result": []}
    return {
        "code": 0,
        "msg": "success",
        "result": result,
        "config": config.copy()
    }



if __name__ == "__main__":
    test()
