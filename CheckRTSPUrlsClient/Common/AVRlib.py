#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import socket
import psutil
import signal
import platform
import time
import Common.HoonUtils as utils
import json
import moviepy.editor as mpy
from enum import Enum
import numpy as np
import more_itertools as itools
import cv2
import functools
import traceback

os_type = platform.system()
if os_type == "Darwin":
    START_PROC_PREFIX  = " nohup "
    START_PROC_POSTFIX = " > /dev/null 2>&1 & "
elif os_type == "Linux":
    START_PROC_PREFIX  = " nohup "
    START_PROC_POSTFIX = " > /dev/null 2>&1 & "
    # START_PROC_POSTFIX = " & "
elif os_type == "Windows":
    START_PROC_PREFIX  = " start /min "
    START_PROC_POSTFIX = " "
else:
    START_PROC_PREFIX  = " nohup "
    START_PROC_POSTFIX = " > /dev/null 2>&1 & "

PYTHON_PATH = os.path.dirname(sys.executable)
PATH_SETUP    = "export PATH="            + PYTHON_PATH + ":$PATH"
LIB_SETUP     = "export LIBRARY_PATH="    + PYTHON_PATH + "/../lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
LD_LIB_SETUP  = "export LD_LIBRARY_PATH=" + PYTHON_PATH + "/../lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
ENV_SETUP = PATH_SETUP + ';' + LIB_SETUP + ';' + LD_LIB_SETUP + ';'
MOD_SERVER_ARGS = " --op_mode server --logging "


RESULT_FILE_EXT = '.rst'
VIDEO_FILE_EXT  = '.mp4'
IMAGE_FILE_EXT  = '.jpg'
JSON_FILE_EXT   = '.json'
TXT_FILE_EXT    = '.txt'
MMAP_FILE_EXT   = '.mmap'
AVI_FILE_EXT    = '.avi'

CLIENT_EXIT  = -1
SERVER_START = 0
SERVER_CHECK = 1
SERVER_RUN   = 3
SERVER_STOP  = 4

sVIDEO  = 'VIDEO'
sIMAGE  = 'IMAGE'
sSERVER = 'SERVER'
sBOTH   = 'BOTH'

IMG_ORG_SUB_FOLDER = "img_org"
IMG_VEH_SUB_FOLDER = "img_veh"
IMG_PLT_SUB_FOLDER = "img_plt"
IMG_FACE_SUB_FOLDER = "img_face"
JSON_IMG_SUB_FOLDER = "json_img"
JSON_VEH_SUB_FOLDER = "json_veh"
TXT_VEH_SUB_FOLDER = "txt_veh"
JSON_FACE_SUB_FOLDER = "json_face"


class ModuleOpMode(Enum):
    video  = 0
    image  = 1
    server = 2
    client = 3
    standalone = 4
    test = 5


class ModuleOutputMode(Enum):
    video = 0
    image = 1


class ModuleType(Enum):
    ImgAVR        = 1
    ObjDetect     = 2
    PlateDetect   = 3
    PlateRecog    = 4
    VehicleClass  = 5
    WinDetect     = 6
    FaceDetect    = 7
    RstStorage    = 8
    SocketTx      = 9


ModuleAcronym = ["", "IA", "OD", "PD", "PR", "VC", "WD", "FD", "RS", "ST"]


class ModuleManagerOpMode(Enum):
    stop  = -1
    start = 0
    check = 1
    run   = 2


class SystemInputType(Enum):
    video = '0'
    image = '1'


class BustClassCode(Enum):
    unknown = '00'
    green   = '10'
    namsan  = '20'
    public  = '30'
    jemulpo = '40'


class ObjectType(Enum):
    other   = 0
    car     = 1
    van     = 2
    bus     = 3
    truck   = 4
    plate   = 5
    window  = 6


class VehicleSide(Enum):    # 차량의 전면/후면
    unknown = 0
    front   = 1
    back    = 2


class VehicleDir(Enum):     # 지점을 기준으로 진입/진출
    unknown = 0
    entry   = 1
    exit    = 2


class VehicleType(Enum):
    unknown    = 0
    car        = 1
    van        = 2
    bus        = 3
    truck      = 4


class VehicleColor(Enum):
    unknown  = 0
    white    = 1
    silver   = 2
    gray     = 3
    black    = 4
    red      = 5
    darkblue = 6
    blue     = 7
    yellow   = 8
    green    = 9
    brown    = 10
    pink     = 11
    purple   = 12
    darkgray = 13
    cyan     = 14


class LaneInfo(Enum):
    unknown = 0
    lane1   = 1
    lane2   = 2
    lane3   = 3
    lane4   = 4
    lane5   = 5
    lane6   = 6
    lane7   = 7
    lane8   = 8
    lane9   = 9


class ManagerCode(Enum):
    unknown = '0'
    green   = '1'     # 녹색교통
    crowded = '2'     # 혼잡


class CreatorCode(Enum):
    ai      = 'AA'
    anpr    = 'AN'
    tunnel1 = 'NO'
    tunnel3 = 'NT'


class BustPurposeCode(Enum):
    green = 'GG'
    toll  = 'TT'
    park  = 'PP'


class ImageFeature:

    def __init__(self, dict_dat=None):

        self.camera_id = "123456789"    # 카메라 ID (9 자리)
        self.taken_date = None          # 통과시간, 날짜 (8 자리)
        self.taken_time = None          # 통과시간, 시간 (8 자리)
        self.uri = None                 # 전체 이미지 경로
        self.input_type = None          # 입력 종류
        self.bust_class = None          # 단속대상 구분

        self.manager = None             # 관리주체 코드 (1 자리)
        self.creator = "AA"             # 생성기관 코드 (2 자리)
        self.bust_purpose = None        # 단속 목적 (2 자리)
        self.desc = None                # 이미지 설명 (1 자리)
        self.serial_num = None          # 생성 기관이 부여하는 개별 일련번호 (8 자리)

        self.mmap_fname = None
        self.mmap_shape = None
        self.roi = None

        '''
        self.roi_mmap_fname = None
        self.roi_mmap_shape = None
        self.roi_roi = None
        self.roi_offset = None
        '''

        self.name = None
        self.proc_time = None

        self.veh_pos_arr = None
        self.veh_conf_arr = None
        self.win_pos_arr = None
        self.win_conf_arr = None
        self.plt_pos_arr = None
        self.plt_num_arr = None
        self.plt_conf_arr = None
        self.plt_type_arr = None
        self.face_pos_arr = None
        self.face_conf_arr = None

        self.obj_arr = None

        if dict_dat:
            # self.put_dict_dat(dict_dat)
            self.init_with_dict(dict_dat)

    def init_with_dict(self, dict_dat):
        for key in dict_dat:
            if hasattr(self, key):
                setattr(self, key, dict_dat[key])

    def put_dict_dat(self, dict_dat):

        self.camera_id = dict_dat["camera_id"]
        self.taken_date = dict_dat["taken_date"]
        self.taken_time = dict_dat["taken_time"]
        self.uri = dict_dat["uri"]
        self.input_type = dict_dat["input_type"]
        self.bust_class = dict_dat["bust_class"]

        self.manager = dict_dat["manager"]
        self.creator = dict_dat["creator"]
        self.bust_purpose = dict_dat["bust_purpose"]
        self.desc = dict_dat["desc"]
        self.serial_num = dict_dat["serial_num"]

        self.mmap_fname = dict_dat["mmap_fname"]
        self.mmap_shape = dict_dat["mmap_shape"]
        self.roi = dict_dat["roi"]

        '''
        self.roi_mmap_fname = dict_dat["roi_mmap_fname"]
        self.roi_mmap_shape = dict_dat["roi_mmap_shape"]
        self.roi_roi = dict_dat["roi_roi"]
        self.roi_offset = dict_dat["roi_offset"]
        '''

        self.name = dict_dat["name"]
        self.proc_time = dict_dat["proc_time"]

        self.veh_pos_arr = dict_dat["veh_pos_arr"]
        self.veh_conf_arr = dict_dat["veh_conf_arr"]
        self.win_pos_arr = dict_dat["win_pos_arr"]
        self.win_conf_arr = dict_dat["win_conf_arr"]
        self.plt_pos_arr = dict_dat["plt_pos_arr"]
        self.plt_num_arr = dict_dat["plt_num_arr"]
        self.plt_conf_arr = dict_dat["plt_conf_arr"]
        self.plt_type_arr = dict_dat["plt_type_arr"]
        self.face_pos_arr = dict_dat["face_pos_arr"]
        self.face_conf_arr = dict_dat["face_conf_arr"]
        self.obj_arr = dict_dat["obj_arr"]

    def get_dict_dat(self):
        return {
            "camera_id": self.camera_id,
            "taken_date": self.taken_date,
            "taken_time": self.taken_time,
            "uri": self.uri,
            "input_type": self.input_type,
            "bust_class": self.bust_class,

            "manager": self.manager,
            "creator": self.creator,
            "desc": self.desc,
            "bust_purpose": self.bust_purpose,
            "serial_num": self.serial_num,

            "mmap_fname": self.mmap_fname,
            "mmap_shape": self.mmap_shape,
            "roi": self.roi,

            '''
            "roi_mmap_fname": self.roi_mmap_fname,
            "roi_mmap_shape": self.roi_mmap_shape,
            "roi_roi" : self.roi_roi,
            "roi_offset": self.roi_offset,
            '''

            "name": self.name,
            "proc_time": self.proc_time,

            "veh_pos_arr": self.veh_pos_arr,
            "veh_conf_arr": self.veh_conf_arr,
            "win_pos_arr": self.win_pos_arr,
            "win_conf_arr": self.win_conf_arr,
            "plt_pos_arr": self.plt_pos_arr,
            "plt_num_arr": self.plt_num_arr,
            "plt_conf_arr": self.plt_conf_arr,
            "plt_type_arr": self.plt_type_arr,
            "face_pos_arr": self.face_pos_arr,
            "face_conf_arr": self.face_conf_arr,
            "obj_arr": self.obj_arr,
        }

    def set_datetime(self, dt):
        self.taken_date = dt.strftime("%Y%m%d")[:8]
        self.taken_time = dt.strftime("%H%M%S%f")[:8]

    def get_roi_img(self):
        img = np.memmap(self.mmap_fname, dtype='uint8', mode='r', shape=tuple(self.mmap_shape))
        box = [[min(self.roi[0][0], self.roi[2][0]), min(self.roi[0][1], self.roi[1][1])],
               [max(self.roi[1][0], self.roi[3][0]), max(self.roi[2][1], self.roi[3][1])]]
        return img[box[0][1]:box[1][1], box[0][0]:box[1][0]], box[0]

    def imshow_img(self):
        img = np.memmap(self.mmap_fname, dtype='uint8', mode='r', shape=tuple(self.mmap_shape))
        utils.imshow(img)


class VehicleFeature:

    def __init__(self, dict_dat=None):

        self.type = VehicleType.unknown        # 차량종류. ObjectDetect(또는 Vehicle Detect)에서 기입한다.
        self.color = VehicleColor.unknown      # 차량색상. ObjectDetect(또는 Vehicle Detect)에서 기입한다.
        self.veh_uri = None                    # 차량 이미지 경로(마스킹 포함). veh_pos를 사용해서 RstStorage에서 기입한다.
        self.veh_side = VehicleSide.unknown    # 차량 전면/후면. ObjectDetect(또는 Vehicle Detect)에서 기입한다.
        self.veh_dir = VehicleDir.unknown      # 지점 기준 진입/진출. veh_side과 DB의 진/출입 정보를 사용하여 ImgAVR 또는 RstStorage에서 결정
        self.lane_info = LaneInfo.unknown      # 차선정보. ImgAVR 또는 RstStorage에서 결정
        self.proc_time = None                  # 인식소요시간. ImgAVR에서 기입한다.
        self.plt_uri = None                    # 번호판 이미지 경로. plt_pos를 사용해서 RstStorage에서 기입한다.

        self.mmap_fname = None
        self.mmap_shape = None
        self.roi = None

        self.veh_pos = None
        self.veh_conf = None
        self.win_pos = None
        self.win_conf = None
        self.plt_pos = None
        self.plt_num = None                         # 차량번호
        self.plt_conf = None                        # 인식신뢰도
        self.plt_type = None
        self.face_pos_arr = None
        self.face_conf_arr = None

        self.number = None
        if dict_dat:
            self.put_dict_dat(dict_dat)

    def put_dict_dat(self, dict_dat):

        self.type = VehicleType(dict_dat["type"])
        self.color = VehicleColor(dict_dat["color"])
        self.veh_uri = dict_dat["veh_uri"]
        self.veh_side = VehicleSide(dict_dat["veh_side"])
        self.veh_dir = VehicleDir(dict_dat["veh_dir"])
        self.lane_info = LaneInfo(dict_dat["lane_info"])
        self.proc_time = dict_dat["proc_time"]
        self.plt_uri = dict_dat["plt_uri"]

        self.mmap_fname = dict_dat["mmap_fname"]
        self.mmap_shape = dict_dat["mmap_shape"]
        self.roi = dict_dat["roi"]

        self.veh_pos = dict_dat["veh_pos"]
        self.veh_conf = dict_dat["veh_conf"]
        self.win_pos = dict_dat["win_pos"]
        self.win_conf = dict_dat["win_conf"]
        self.plt_pos = dict_dat["plt_pos"]
        self.plt_num = dict_dat["plt_num"]
        self.plt_conf = dict_dat["plt_conf"]
        self.plt_type = dict_dat["plt_type"]
        self.face_pos_arr = dict_dat["face_pos_arr"]
        self.face_conf_arr = dict_dat["face_conf_arr"]

    def get_dict_dat(self):
        return {
            "type": self.type.value,
            "color": self.color.value,
            "veh_uri": self.veh_uri,
            "veh_side": self.veh_side.value,
            "veh_dir": self.veh_dir.value,
            "lane_info": self.lane_info.value,
            "proc_time": self.proc_time,
            "plt_uri": self.plt_uri,

            "mmap_fname": self.mmap_fname,
            "mmap_shape": self.mmap_shape,
            "roi" : self.roi,

            "veh_pos": self.veh_pos,
            "veh_conf": self.veh_conf,
            "win_pos": self.win_pos,
            "win_conf": self.win_conf,
            "plt_pos": self.plt_pos,
            "plt_num": self.plt_num,
            "plt_conf": self.plt_conf,
            "plt_type": self.plt_type,
            "face_pos_arr": self.face_pos_arr,
            "face_conf_arr": self.face_conf_arr
        }

    def get_temp_veh_pos(self, plt_pos, x_ratio=5, y_ratio=20, bottom_space_ratio=2):
        img_h, img_w = self.mmap_shape[:2]
        plate_w = plt_pos[1][0] - plt_pos[0][0]
        vehicle_w = int(plate_w * x_ratio)
        vehicle_x = plt_pos[0][0] - (vehicle_w - plate_w) / 2
        vehicle_x = max(int(vehicle_x), 0)
        vehicle_x_2 = min(vehicle_x + vehicle_w, img_w)

        plate_h = plt_pos[3][1] - plt_pos[0][1]
        vehicle_h = int(plate_h * y_ratio)
        vehicle_y = plt_pos[0][1] - (vehicle_h - plate_h * bottom_space_ratio)
        vehicle_y = max(int(vehicle_y), 0)
        vehicle_y_2 = min(vehicle_y + vehicle_h, img_h)

        return [vehicle_x, vehicle_y, vehicle_x_2, vehicle_y_2]


class SocketHeader:

    def __init__(self):

        self.header_type = None
        self.system_code = None
        self.switch_pattern = None
        self.msg_type = None

        self.json_str = None
        self.tot_len = 3 + 3 + 1 + 3

    def set_to_default(self):

        self.header_type = '601'
        self.system_code = 'AIS'
        self.switch_pattern = '1'
        self.msg_type = 'M63'

    def get_header_string(self):

        header_str  = get_string_w_len(self.header_type, 3)
        header_str += get_string_w_len(self.system_code, 3)
        header_str += get_string_w_len(self.switch_pattern, 1)
        header_str += get_string_w_len(self.msg_type, 3)

        return header_str

    @staticmethod
    def int_to_bytes(x: int, length=4, signed: bool = False) -> bytes:
        return x.to_bytes(length, byteorder='big', signed=signed)

    @staticmethod
    def bytes_to_int(x: bytes, signed: bool = False) -> int:
        return int.from_bytes(x, byteorder='big', signed=signed)

    def parse_header_string(self, header_str):

        self.header_type = header_str[:3]
        self.system_code = header_str[3:6]
        self.switch_pattern = header_str[6:7]
        self.msg_type = header_str[7:10]

    def print_header_info(self):

        msg  = "\n * Header information"
        msg += "\n   > header type    = {}".format(self.header_type)
        msg += "\n   > system code    = {}".format(self.system_code)
        msg += "\n   > switch pattern = {}".format(self.switch_pattern)
        msg += "\n   > message type   = {}".format(self.msg_type)
        # msg += "\n   > body length    = {}\n".format(self.body_len)

        return msg


class SocketBody:

    def __init__(self, dict_info=None, json_str=None):

        self.dict = dict_info
        self.json_str = json_str

        if self.dict:
            self.json_str = json.dumps(self.dict)
        if self.json_str:
            self.dict = json.loads(self.json_str)

    def generate_socket_body(self, img_feat, veh_feat):
        if veh_feat.veh_dir == VehicleDir.entry:
            veh_dir = 0
        elif veh_feat.veh_dir == VehicleDir.exit:
            veh_dir = 1
        else:
            veh_dir = 2

        if not veh_feat.plt_conf:
            veh_feat.plt_conf = 0
        elif veh_feat.plt_conf >= 100:
            veh_feat.plt_conf = 99
        plt_conf = "{:02.0f}".format(veh_feat.plt_conf)

        self.dict = {"camId": get_string_w_len(img_feat.camera_id, 9),
                     "pasageTime": get_string_w_len(img_feat.taken_date, 8) +
                                   get_string_w_len(img_feat.taken_time, 6, pre_stuffing=True, logger=None),
                     "vhcleNum": veh_feat.plt_num,
                     "vhcleKnd": get_string_w_len(veh_feat.type.value, 1),
                     "vhcleColor": get_string_w_len(veh_feat.color.value, 2, pre_stuffing=True, stuffing_char='0'),
                     "allImagePath": veh_feat.veh_uri,
                     "vhcleImagePath": "",
                     "nopltImagePath": veh_feat.plt_uri,
                     "inputKind": get_string_w_len(img_feat.input_type, 1),
                     "vhcleDrc": get_string_w_len(veh_dir, 1),
                     "recogCnfdncRate": get_string_w_len(plt_conf, 2, pre_stuffing=True, stuffing_char='0'),
                     "tfclneInfo": get_string_w_len(veh_feat.lane_info.value, 1),
                     "recogReqreTime": get_string_w_len(veh_feat.proc_time, 6, pre_stuffing=True),
                     "regltTrgetSe": get_string_w_len(img_feat.bust_class, 2)}

        self.json_str = json.dumps(self.dict)

        return self.dict,  self.json_str


"""
class FaceFeature:

    def __init__(self, dict_dat=None):

        self.mmap_fname = None
        self.mmap_shape = None
        self.roi = None

        self.face_pos = None
        self.face_conf = None

        if dict_dat:
            self.put_dict_dat(dict_dat)

    def put_dict_dat(self, dict_dat):

        self.mmap_fname = dict_dat["mmap_fname"]
        self.mmap_shape = dict_dat["mmap_shape"]
        self.roi = dict_dat["roi"]

        self.face_pos = dict_dat["face_pos"]
        self.face_conf = dict_dat["face_conf"]

    def get_dict_dat(self):
        return {
            "mmap_fname": self.mmap_fname,
            "mmap_shape": self.mmap_shape,
            "roi" : self.roi,

            "face_pos": self.face_pos,
            "face_conf": self.face_conf
        }
"""

"""
class PlateFeature:

    def __init__(self, dict_dat=None):

        self.mmap_fname = None
        self.mmap_shape = None
        self.roi = None

        self.plt_num = None
        self.plt_type = None
        self.plt_pos = None
        self.plt_conf = None

        if dict_dat:
            self.put_dict_dat(dict_dat)

    def put_dict_dat(self, dict_dat):

        self.mmap_fname = dict_dat["mmap_fname"]
        self.mmap_shape = dict_dat["mmap_shape"]
        self.roi = dict_dat["roi"]

        self.plt_num = dict_dat['plt_num']
        self.plt_type = dict_dat['plt_type']
        self.plt_pos = dict_dat["plt_pos"]
        self.plt_conf = dict_dat['plt_conf']

    def get_dict_dat(self):
        return {
            "mmap_fname": self.mmap_fname,
            "mmap_shape": self.mmap_shape,
            "roi" : self.roi,

            "plt_num": self.plt_num,
            "plt_type": self.plt_type,
            "plt_pos": self.plt_pos,
            "plt_conf": self.plt_conf
        }
"""


class ServerFeature:

    def __init__(self, ini=None):

        self.name = None
        self.ip = None
        self.port = None
        self.mmap_fname = None
        self.acronym = None

        if ini:
            self.init_ini(ini)

    def init_ini(self, ini):

        self.ip = ini['ip']
        self.port = int(ini['port'])

        try:
            self.name = ini['name']
        except KeyError:
            self.name = ''

        try:
            self.mmap_fname = ini['mmap_fname']
        except KeyError:
            self.mmap_fname = ''

        try:
            self.acronym = ini['acronym']
        except KeyError:
            self.acronym = ''


def check_port(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # noinspection PyBroadException
        try:
            s.bind((ip, port))
            return False
        except:
            return True


def kill_process(port, name="", logger=None):
    for proc in psutil.process_iter():
        try:
            for conns in proc.connections(kind='inet'):
                if conns.laddr.port == port:
                    # noinspection PyBroadException
                    try:
                        proc.send_signal(signal.SIGTERM)
                        proc.send_signal(signal.SIGKILL)
                    except:
                        pass
                    if logger:
                        logger.info(" > Killed the process {} using {:d} port\n".format(name, port))
                    time.sleep(1)
        except psutil.AccessDenied:
            pass
        except psutil.ZombieProcess:
            pass
        except Exception as e:
            print(e)


def socket_client_bytes(ip, port, send_dat,
                        logger=utils.get_stdout_logger(),
                        show_send_dat_=False,
                        show_recv_dat_=False,
                        prefix=" #",
                        recv_=True):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, port)

    try:
        if logger:
            logger.info(prefix + " Connecting socket to %s port %s..." % server_address)
        sock.connect(server_address)
    except Exception as e:

        if logger:
            logger.error(e)
        return False, None

    recv_dat = None

    try:

        # sock.sendall(send_dat.encode('utf-8'))
        sock.sendall(send_dat)
        if logger:
            logger.info(prefix + " Sent {} bytes".format(len(send_dat)))
        if show_send_dat_ and logger is not None:
            logger.info(prefix + " Sent data: {}".format(send_dat))

        if recv_:
            recv_dat = utils.recv_all(sock, logger=logger if show_recv_dat_ else None).decode('utf-8')
            if not recv_dat:
                if logger:
                    logger.error(prefix + " failed to received")
                return False, None
            if logger:
                logger.info(prefix + " Received {} bytes".format(len(recv_dat)))
            if show_recv_dat_ and logger is not None:
                logger.info(prefix + " Received data:  {}".format(recv_dat))
        else:
            recv_dat = {'state': "Done"}
    except Exception as e:
        if logger:
            logger.error(e)
        return False, recv_dat
    finally:
        if logger:
            logger.info(prefix + " Closing socket.")
        sock.close()
        return True, recv_dat


def socket_client(ip, port, send_dat,
                  logger=utils.get_stdout_logger(),
                  show_send_dat_=False,
                  show_recv_dat_=False,
                  prefix=" #",
                  recv_=True):
    return socket_client_bytes(ip, port, send_dat.encode('utf-8'),
                               logger=logger,
                               show_send_dat_=show_send_dat_,
                               show_recv_dat_=show_recv_dat_,
                               prefix=prefix,
                               recv_=recv_)


def handle_command_request(sock, func=None, response_=True, logger=utils.get_stdout_logger()):

    logger.info('')
    logger.info(' # Waiting for a connection...')

    con, client_address = sock.accept()
    logger.info(" # Connected with {} at {}.".format(client_address, utils.get_datetime()[:-3]))

    ret_ = True
    sent_msg = ''
    dict_dat = {}
    try:
        str_dat = utils.recv_all(con, recv_buf_size=4092, logger=None).decode('utf-8')
        logger.info(" # Received {:d} bytes.".format(len(str_dat)))
        # logger.info(" # Received: \"{}\"".format(str_dat))
        if utils.is_json(str_dat):
            dict_dat = json.loads(str_dat)
            cmd = dict_dat['cmd'].lower()
            if cmd == 'check':
                logger.info(" # Received \"check\" command")
                sent_msg = '{"state":"healthy"}'
            elif cmd == 'stop':
                logger.info(" # Received \"stop\" command")
                sent_msg = '{"state":"Bye"}'
                ret_ = False
            elif cmd == 'run':
                if func:
                    stt_time = time.time()
                    resp = func(dict_dat['request'], logger=logger)
                    proc_time = time.time() - stt_time
                else:
                    resp, proc_time = None, 0
                sent_msg = json.dumps({"state": "Done", "response": resp, "proc_time": proc_time})
            else:
                logger.error(" @ Invalid command, {}.".format(cmd))
                sent_msg = '{"state":"Invalid"}'
        else:
            sent_msg = '{"state":"Not json"}'
    except Exception as e:
        logger.error(str(e) + "\n" + traceback.format_exc())
        sent_msg = '{"state":"' + str(e) + '"}'

    finally:
        if response_:
            con.sendall(sent_msg.encode('utf-8'))
            logger.info(" # Sent: {:d} bytes, {}".format(len(sent_msg), sent_msg))
        con.close()

    return ret_, dict_dat


def create_video_from_images(img_arr, vid_fname, duration=2, fps=30):
    clips = [mpy.ImageClip(m).set_duration(duration) for m in img_arr]
    concat_clip = mpy.concatenate_videoclips(clips, method='compose')
    concat_clip.write_videofile(vid_fname, fps=fps)
    return True


def save_video_file_from_images(img_arr, vid_fname,
                                duration=2, fps=30,
                                logger=utils.get_stdout_logger()):
    clips = [mpy.ImageClip(m).set_duration(duration) for m in img_arr]
    concat_clip = mpy.concatenate_videoclips(clips, method='compose')
    concat_clip.write_videofile(vid_fname, fps=fps)
    logger.info(" # save video file from {:d} images, {}.".format(len(img_arr), vid_fname))
    return True


def get_server_socket(ip, port,
                      logger=utils.get_stdout_logger(), proc_name='', listen_num=5):

    logger.info(" # Getting server socket...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, port)
    # logger.info(server_address)
    # check = check_port(ip, port, logger=logger)
    # logger.info(check)
    # if check:
    if check_port(ip, port):    # , logger=logger):
        logger.info(" # Port, {:d}, was already taken. "
                    "The process using {:d} will be killed first.".format(port, port))
        kill_process(port, name=proc_name)

    logger.info(" # Starting up \"{}\" SERVER on {}:{:d}...".format(proc_name, ip, port))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(server_address)
    sock.listen(listen_num)

    return sock


def save_video_from_image_array(img_arr, base_fname, out_path=None,
                                vid_duration=2, logger=utils.get_stdout_logger()):

    if vid_duration <= 0 or not img_arr:
        return

    out_fname = os.path.join(out_path, base_fname) if out_path else base_fname
    out_fname += VIDEO_FILE_EXT

    logger.info(" # save video from {:d} images, {}.".format(len(img_arr), out_fname))
    create_video_from_images(img_arr, out_fname, duration=vid_duration)


def save_txt_file(contents, fname, out_path=None, desc='', logger=utils.get_stdout_logger()):

    txt_fname = os.path.join(out_path, fname) if out_path else fname
    logger.info(" # save {} file, {}.".format(desc, txt_fname))
    with open(txt_fname, "w") as f:
        f.write(contents)


def get_images_from_video(vid_fname, out_path, frame_interval, logger=utils.get_stdout_logger()):

    utils.file_exists(vid_fname, exit_=True)
    utils.folder_exists(out_path, exit_=False, create_=True, print_=True)

    logger.info(" # Extract image from video, {}".format(vid_fname))

    vid = mpy.VideoFileClip(vid_fname)
    base_fname = os.path.splitext(os.path.basename(vid_fname))[0]
    i_digit = int(np.log10(vid.duration / frame_interval)) + 1
    n_digit = int(np.log10(vid.duration)) + 3

    for i, s in enumerate(itools.numeric_range(0, vid.duration, frame_interval)):

        frame = vid.get_frame(s)
        time_info = "__" + \
                    "{:0{width}d}".format(i, width=i_digit) + \
                    "__" + \
                    "{:0{width}.1f}sec".format(s, width=n_digit)
        out_fname = os.path.join(out_path, base_fname + time_info + IMAGE_FILE_EXT)
        utils.imwrite(frame, out_fname)
        logger.info(" # save image, {}".format(out_fname))


def save_dict_to_json_file(dict_dat, json_fname, logger=utils.get_stdout_logger()):
    with open(json_fname, "w") as f:
        json.dump(dict_dat, f)
    logger.info(" # Save dict to json file, {}".format(json_fname))


def check_module_servers(server_features,
                         exit_=False,
                         show_send_dat_=False,
                         show_recv_dat_=False,
                         logger=utils.get_stdout_logger()):

    send_dat = json.dumps({"cmd": "check"})

    fail_list = []

    for server_feat in server_features:

        if isinstance(server_features, list):
            ip = server_feat.ip
            port = server_feat.port
            # name = server_feat.name
            acronym = server_feat.acronym
        elif isinstance(server_features, dict):
            ip = server_features[server_feat].ip
            port = server_features[server_feat].port
            # name = server_features[server_feat].name
            acronym = server_features[server_feat].acronym
        else:
            ip, port, name, acronym = None, None, None, None

        if check_port(ip, port):
            if socket_client(ip, port, send_dat,
                             logger=None, show_send_dat_=show_send_dat_,
                             show_recv_dat_=show_recv_dat_, prefix=" #check#")[0]:
                logger.info(" # {}://{}:{:d} is healthy.".format(acronym, ip, port))
            else:
                logger.error(" @ Error: {}://{}:{:d} is NOT healthy.".format(acronym, ip, port))
                fail_list.append([ip, port])
        else:
            logger.info(" @ Error in module server, {}://{}:{:d}.".format(acronym, ip, port))
            fail_list.append([ip, port])

    if len(fail_list) > 0:
        if exit_:
            sys.exit(1)

    return len(fail_list) == 0, fail_list


def get_idx_of_server_features_by_module_name(server_features, mod_name):
    return [x.name for x in server_features].index(mod_name)


def send_run_request_and_recv_response(ip, port, request,
                                       show_send_dat_=True,
                                       show_recv_dat_=True,
                                       exit_=False,
                                       desc="",
                                       recv_=True,
                                       logger=utils.get_stdout_logger()):

    send_dat = {"cmd": "run", "request": request}
    ret, recv_dat = socket_client(ip, port, json.dumps(send_dat),
                                  show_send_dat_=show_send_dat_,
                                  show_recv_dat_=show_recv_dat_,
                                  prefix=desc,
                                  recv_=recv_,
                                  logger=logger)

    if not ret:
        logger.error(" @ Error in process request and response method : Socket failed. port={}".format(port))
        if exit_:
            sys.exit(1)
        else:
            return None, None

    if recv_:
        assert(recv_dat is not None)

        try:
            recv_dict = json.loads(recv_dat)
        except Exception as e:
            logger.error("{} @ Error: load json : {}".format(desc, e))
            if exit_:
                sys.exit(1)
            else:
                return None, None

        # if 'response' not in recv_dict:
        #     a = 2
        resp_dict = recv_dict['response']
        proc_time = recv_dict['proc_time']

        if not recv_dict['state'] == "Done":
            logger.error(" @ Error in process request and response method : Response failed. port={}".format(port))
            if exit_:
                sys.exit(1)
            else:
                return None, None

        return resp_dict, proc_time
    else:
        return recv_dat, None


def show_message(logger, msg, postfix="\n"):
    logger.info(msg)
    return msg + postfix


def get_string_w_len(x, length, pre_stuffing=False, stuffing_char=' ',
                     logger=utils.get_stdout_logger()):

    xx = str(x)
    x_len = len(xx)

    if x_len < length:

        if pre_stuffing:
            out_str = stuffing_char[0] * (length - x_len) + xx
        else:
            out_str = xx + stuffing_char[0] * (length - x_len)

    elif x_len > length:

        if pre_stuffing:
            out_str = xx[:length]
        else:
            out_str = xx[x_len-length:]

        if logger:
            logger.error(" @ Error: variable length, {:d}, is beyond the definition, {:d}.".
                         format(x_len, length))
    else:
        out_str = xx

    return out_str


def get_roi(img, roi, imshow_sec=-1, clockwise_=True):

    roi = np.array(roi)
    roi = roi * np.array(img.shape[1::-1]) if not sum(sum(roi > 1)) else roi
    if clockwise_:
        roi[[2,3]] = roi[[3,2]]

    roi_corners = np.array([[tuple(x) for x in roi]],dtype=np.int32)
    ignore_mask_color = (255,) * img.shape[2]
    mask = cv2.fillPoly(np.zeros(img.shape, dtype=np.uint8), roi_corners, color=ignore_mask_color)
    roi_img = cv2.bitwise_and(img, mask)
    utils.imshow(roi_img, desc="roi image", pause_sec=imshow_sec)

    # gray_img = cv2.cvtColor(roi_img, code=cv2.COLOR_RGB2GRAY)
    offset = [[0,0], list(img.shape[1::-1])]

    return roi_img, offset


def norm_rect(rect, max_shape):
    return [0 if rect[0] < 0 else rect[0],
            0 if rect[1] < 0 else rect[1],
            max_shape[0] if rect[2] > max_shape[0] else rect[2],
            max_shape[1] if rect[3] > max_shape[1] else rect[3]]


def start_mod_server(ip,port, mod_name, ini_fname, wait_period=10, logger=utils.get_stdout_logger()):

    if check_port(ip, port):
        logger.info(" # {}://{}:{:d} is already running. Run after killing it".format(mod_name, ip, port))
        kill_mod_server(ip, port, mod_name=mod_name, logger=logger)

    py_fname = os.path.join("..", mod_name, mod_name + ".py")
    os.system(ENV_SETUP +
              START_PROC_PREFIX +
              " python -O  " + py_fname +
              " --ini " + ini_fname +
              MOD_SERVER_ARGS +
              START_PROC_POSTFIX)
    time.sleep(wait_period)
    logger.info(" # {}://{}:{:d} has launched.".format(mod_name, ip, port))


def check_mod_server(ip, port, mod_name="NoName", prefix="", debug_=True, logger=utils.get_stdout_logger()):

    check_rst = False
    if check_port(ip, port):
        send_dat = json.dumps({"cmd": "check"})
        ret, recv_dat = socket_client(ip, port, send_dat,
                                      show_send_dat_=debug_,
                                      show_recv_dat_=debug_,
                                      prefix=prefix,
                                      logger=logger if debug_ else None)

        if ret and len(recv_dat)>0:
            res = json.loads(recv_dat)
            if "state" in res and res["state"] == "healthy":
                check_rst = True
    else:
        logger.info(" @ {}://{}:{:d} DOES NOT exist.".format(mod_name, ip, port))
        return check_rst

    if check_rst:
        logger.info(" # {}://{}:{:d} is in healthy state.".format(mod_name, ip, port))
    else:
        logger.info(" # {}://{}:{:d} is NOT in healthy state.".format(mod_name, ip, port))

    return check_rst


def check_mod_server_and_start(ip,
                               port,
                               mod_name="NoName",
                               mod_ini_fname=None,
                               trial_num=1,
                               prefix="",
                               logger=utils.get_stdout_logger()):

    ret = False
    for i in range(trial_num):

        if check_mod_server(ip, port, mod_name=mod_name, prefix=prefix, logger=logger):
            ret = True
            break
        if i == trial_num - 1:
            break

        logger.info("{} Try to start {}//{}:{:d}.".format(prefix, mod_name, ip, port))
        start_mod_server(ip, port, mod_name, mod_ini_fname, logger=logger)

    return ret


def kill_mod_server(ip, port, mod_name="NoName", wait_period=3, logger=utils.get_stdout_logger()):

    if check_port(ip, port):
        kill_process(port, name=mod_name)
        time.sleep(wait_period)
        logger.info(" # {}://{}:{:d}, was killed.".format(mod_name, ip, port))
    else:
        logger.info(" # {}://{}:{:d} DOES NOT exist.".format(mod_name, ip, port))


def put_roi_to_db_format(quad, width, height):
    quad_norm = [[int(p[0]/width * 1000)/1000., int(p[1]/height*1000)/1000.] for p in quad]
    return ','.join(map(str, functools.reduce(lambda x,y: x+y, quad_norm)))


def get_roi_from_db_format(roi_str, width, height):
    quad_norm = np.array([float(x) for x in roi_str.split(',')]).reshape((4, 2)).tolist()
    return [[p[0]*width, p[1]*height] for p in quad_norm]
