import base64
import time
import requests

from ppadb.client import Client

from adbinit import init_adb, enable_tcpip, restart_server

ocr_url = "http://192.168.3.5:8000/model/ocr"
config_url = "http://192.168.3.5:8000/config"

NULL = 0
XIANJIN = 1
YOUQI = 2
DINGBAN = 3
BAOXIANG = 4

def dump_screenshot(device):
    screenshot = device.screencap()
    with open("test3.png", "wb") as f:
        f.write(screenshot)

def get_box_center(box):
    up = max(box[0][1], box[1][1])
    down = min(box[2][1], box[3][1])
    left = max(box[0][0], box[3][0])
    right = min(box[1][0], box[2][0])
    hcenter = (up + (down - up) // 2)
    wcenter = (left + (right - left) // 2)
    return wcenter, hcenter

def weakup_car_center(cancel):
    hcenter = cancel[1] - 220
    wcenter = cancel[0] + 58
    return wcenter, hcenter

def order2str(order):
    name_map = {
        YOUQI: "油漆桶",
        DINGBAN: "钉板",
        XIANJIN: "现金",
        BAOXIANG: "宝箱",
        NULL: "空"
    }
    return name_map[order]

def get_order_type(msgs):
    for msg in msgs:
        box, text = msg
        if "贵" in text[0] or "宾" in text[0]:
            return BAOXIANG
        if "油" in text[0] or "漆" in text[0] or "桶" in text[0]:
            return YOUQI
        if "钉" in text[0] or "板" in text[0]:
            return DINGBAN
        if "现金" in text[0] or "比兹" in text[0] or "金" in text[0] or "兹" in text[0] or "点数" in text[0]:
            return XIANJIN
    return NULL

def weakup_car(msgs, device):
    for msg in msgs:
        box, text = msg
        if "取消" in text[0]:
            device.input_tap(*weakup_car_center(get_box_center(box)))
            return

def get_reward(msgs, device):
    for msg in msgs:
        box, text = msg
        if "收取" in text[0]:
            device.input_tap(*get_box_center(box))
            return

def get_next_center(accept):
    hcenter = accept[1] + 100
    wcenter = accept[0] + 389
    return wcenter, hcenter

def get_refuse_center(accept):
    hcenter = accept[1] + (1756 - 1744)
    wcenter = accept[0] - (523 - 207)
    return wcenter, hcenter

def select_next_order(msgs, device):
    accept_center = None
    for msg in msgs:
        box, text = msg
        if "接受" in text[0]:
            accept_center = get_box_center(box)
            break
    print(f"接受下一个订单")
    device.input_tap(*get_next_center(accept_center))
    time.sleep(0.5)
    device.input_tap(*accept_center)

def refuse_order(msgs, device):
    accept_center = None
    for msg in msgs:
        box, text = msg
        if "接" in text or "受" in text[0]:
            accept_center = get_box_center(box)
            break
    if accept_center is None:
        return
    print(f"拒绝订单!")
    refuse_center = get_refuse_center(accept_center)
    next_center = get_next_center(accept_center)
    device.input_tap(*refuse_center)
    time.sleep(0.5)
    device.input_tap(*next_center)
    time.sleep(0.5)
    device.input_tap(*refuse_center)
    time.sleep(0.5)
    device.input_tap(*next_center)
    time.sleep(0.5)
    device.input_tap(*refuse_center)

def select_current_order(msgs, device):
    print(f"接受当前订单")
    for msg in msgs:
        box, text = msg
        if "接受" in text[0]:
            device.input_tap(*get_box_center(box))
            return

def get_order(msgs, device):
    for msg in msgs:
        box, text = msg
        if "获取" in text[0]:
            device.input_tap(*get_box_center(box))
            return

def check_warning(msgs):
    for msg in msgs:
        box, text = msg
        if "！" in text[0]:
            return True
    return False

def get_baoxiang_reward(result, device):
    for i, res0 in enumerate(result):
        box, text0 = res0
        if "跳过" in text0[0]:
            device.input_tap(*get_box_center(box))

        if "奖励概要" in text0[0]:
            for res1 in result[i:]:
                box, text1 = res1
                if "领取" in text1[0]:
                    device.input_tap(*get_box_center(box))
                    print("领取宝箱奖励")


def check_in_order(result):
    for res in result:
        box, text = res
        if "订单状态" in text[0]:
            return True
        if "订状态" in text[0]:
            return True
        if "产品总数" in text[0]:
            return True
        if "循环订单" in text[0]:
            return True
        if "预估价值" in text[0]:
            return True
    return False

def jump_baoxiang(device):
    time.sleep(0.5)
    device.input_tap(884, 320)
    time.sleep(1)
    device.input_tap(541, 1856)



def get_config(sess):
    response = sess.get(config_url)
    if response.status_code != 200:
        return {
            "only_treasure_chest": False,
            "jump_paint": False,
            "jump_nail_board": False,
            "close_service": False,
            "refresh_time": 1.0,
            "adb_ip": "192.168.3.206",
            "adb_port": 5555
        }
    return response.json()


def service():
    sess = requests.Session()
    config = get_config(sess)
    if config["network_adb"]:
        ip, port = config["adb_ip"], config["adb_port"]
        if not init_adb(ip, port):
            enable_tcpip(port)
            if not init_adb(ip, port):
                restart_server()
                if not init_adb(ip, port):
                    print(f"connect {ip}:{port} failed!")
                    return

    adb = Client(host="127.0.0.1", port=5037)
    devices = adb.devices()

    if len(devices) == 0:
        print(f"not have adb device error!")
        return

    device = devices[0]

    while True:
        if config["close_service"]:
            print("已关闭服务!")
            config = get_config(sess)
            if config["close_service"]:
                time.sleep(2)
                continue

        screenshot = device.screencap()
        data = {
            "image": base64.b64encode(screenshot).decode(),
            "box_threshold": 0.0
        }

        try:
            response = sess.post(ocr_url, json=data)
        except Exception as e:
            time.sleep(2)
            print(f"OCR server error {e} !")
            continue

        if response.status_code != 200:
            time.sleep(2)
            print("OCR server error!")
            continue

        response = response.json()
        
        if response["code"] != 0:
            time.sleep(2)
            print("OCR server error!")
            continue

        config = response["config"]

        if config["close_service"]:
            time.sleep(2)
            continue

        result = response["result"]
        if not check_in_order(result):
            get_baoxiang_reward(result, device)
            time.sleep(2)
            print(f"非订单界面! {response}")
            continue

        for i, res in enumerate(result):
            box, text = res
            if "交货中" in text[0]:
                msgs = result[i:i+10]
                weakup_car(msgs, device)
                order = get_order_type(msgs)
                print(f"\r交货中: {order2str(order)}...", end="")
                break

            if "已完成" in text[0]:
                msgs = result[i:i+10]
                order = get_order_type(msgs)
                get_reward(msgs, device)
                if order == BAOXIANG:
                    jump_baoxiang(device)
                print(f"\r已完成: {order2str(order)}    ")
                break

            if "新的订单" in text[0]:
                msgs = result[i:]
                order_type = get_order_type(msgs)
                if order_type == NULL:
                    print(f"\r等待订单...", end="")
                else:
                    print(f"\r新的订单:", order2str(order_type))
                if order_type == NULL:
                    get_order(msgs, device)
                elif order_type == BAOXIANG:
                    select_current_order(msgs, device)
                elif order_type == XIANJIN:
                    if config["only_treasure_chest"]:
                        refuse_order(msgs, device)
                    else:
                        select_current_order(msgs, device)
                elif order_type == YOUQI:
                    if config["only_treasure_chest"]:
                        refuse_order(msgs, device)
                    elif config["jump_paint"]:
                        select_next_order(msgs, device)
                    else:
                        select_current_order(msgs, device)
                elif order_type == DINGBAN:
                    if config["only_treasure_chest"]:
                        refuse_order(msgs, device)
                    elif config["jump_nail_board"]:
                        select_next_order(msgs, device)
                    else:
                        select_current_order(msgs, device)
                break
        time.sleep(config["refresh_time"])

if __name__ == '__main__':
    service()
