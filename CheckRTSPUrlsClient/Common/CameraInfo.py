import os
import pandas as pd

from Common.AVRlib import *


def extract_camera_id(rtsp_url):
    start_p = rtsp_url.rfind('/')
    if start_p >= 0:
        rtsp_url = rtsp_url[start_p + 1:]

    if len(rtsp_url) > 9:
        rtsp_url = rtsp_url[:9]
    return rtsp_url


# 나중엔 True가 되어야 함. 그리고, ImgAVR에선 단방향이고 unknown 이면 보내지 말아야 함(?)
# 현재는 단방향(진입 또는 진출)이어도 모두 보냄.
use_defined_dir = False


class VehInOutInfo:
    def __init__(self, veh_dir_info, veh_side_info):
        self.dir_dict = dict()

        veh_dir_info = int(veh_dir_info)    # 진출입 구분(00:진입, 01:진출, 02:양방향)
        if veh_dir_info > 2:
            veh_dir_info = 0

        veh_side_idx = int(veh_side_info) - 1  # 전후면구분 (01:전면, 02:후면)
        if veh_side_idx < 0 or veh_side_idx > 1:
            veh_side_idx = 0

        if veh_dir_info < 2:   # 0:진입, 1:진출
            veh_dir = VehicleDir(veh_dir_info+1)
            self.dir_dict[veh_side_idx + 1] = veh_dir
            if not use_defined_dir:
                self.dir_dict[(veh_side_idx + 1) % 2 + 1] = (veh_dir.value + 1) % 2

        else:   # 양방향
            self.dir_dict[veh_side_idx + 1] = VehicleDir.entry
            self.dir_dict[(veh_side_idx + 1) % 2 + 1] = VehicleDir.exit

    def __str__(self):
        return str(self.dir_dict)

    def get_veh_dir(self, veh_side, logger):
        """
        logger.error(" get_veh_dir. {} is not in {}".format(veh_side, self.dir_dict.keys()))
        if len(self.dir_dict.keys()) == 1:  # 단방향(진입/진출)인 경우 고정된걸 리턴한다.
            return list(self.dir_dict.values())[0]
        return VehicleDir.unknown
        """
        if veh_side in self.dir_dict.keys():
            return self.dir_dict[veh_side]

        return VehicleDir.unknown


class CameraInfo:
    def __init__(self, camera_id, rtsp_url, fps, roi,
                 start_lane=1, lane_count=1,
                 veh_dir_info="00", veh_side_info="01"):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.roi = roi

        veh_dir_info = int(veh_dir_info)
        if veh_dir_info == 2:    # 양방향인 경우 1차로로 지정한다.
            self.start_lane = 1
        else:
            self.start_lane = start_lane
        self.lane_count = lane_count

        self.inout_info = VehInOutInfo(veh_dir_info, veh_side_info)

    def __str__(self):
        return "id:{}, url:{}, dir_dict:{}, fps:{}, roi:{}".format(self.camera_id,
                                                                   self.rtsp_url,
                                                                   self.inout_info,
                                                                   self.fps,
                                                                   self.roi)


class CameraInfoLoader:
    def __init__(self, logger, ini=utils.get_ini_parameters("../AVRServer/AVRServer.ini")):
        self.ini = ini
        self.logger = logger
        self.camera_info_dict = dict()

    @staticmethod
    def get_use_camera_id_list(ini, logger):
        use_camera_id_list_file = os.path.join( ini['SYNC_DB']['repo_dir'],
                                                ini['SYNC_DB']['use_camera_id_list_file'] )
        if not os.path.isfile(use_camera_id_list_file):
            logger.error(" # No file : {}".format(use_camera_id_list_file))
            return None

        with open(use_camera_id_list_file, 'r') as f:
            camera_id_list = f.read().splitlines()

        return camera_id_list

    def configure_with_download_list(self, only_camera_for_this_server=True):
        if only_camera_for_this_server:
            camera_id_list = self.get_use_camera_id_list(self.ini, self.logger)
            if not camera_id_list:
                self.camera_info_dict = dict()
                return False
        else:
            camera_id_list = None

        camera_info_list_file, camera_setup_list_file = CameraInfoLoader.load_camera_infos_list_filename(self.ini)
        self.camera_info_dict = self.load_use_camera_info(camera_info_list_file, camera_setup_list_file,
                                                          camera_id_list, True, self.logger)
        return True

    def configure_with_url_list(self, url_list):
        self.logger.info("configure_with_url_list\n{}".format(url_list))
        camera_info_list_file, camera_setup_list_file = CameraInfoLoader.load_camera_infos_list_filename(self.ini)
        self.camera_info_dict = self.load_use_camera_info(camera_info_list_file, camera_setup_list_file,
                                                          url_list, False, self.logger)

    def get_camera_info(self, camera_id):
        if camera_id in self.camera_info_dict.keys():
            return self.camera_info_dict[camera_id]
        return None

    def get_camera_list(self):
        return self.camera_info_dict.keys()

    @staticmethod
    def load_camera_infos_list_filename(ini):
        repo_dir = ini['SYNC_DB']['repo_dir']
        # CAMERA_ID,RTSP_URI,START_LANE,LANE_COUNT,INSTL_DIR,VEH_SIDE
        camera_info_list_file = os.path.join(repo_dir, ini['SYNC_DB']['camera_info_list_file'])
        # CAMERA_ID,ROI,FPS,ALLOC_SERVER_IP
        camera_setup_list_file = os.path.join(repo_dir, ini['SYNC_DB']['camera_setup_list_file'])
        return camera_info_list_file, camera_setup_list_file

    @staticmethod
    def load_camera_info(camera_info_list_file, camera_setup_list_file, camera_id, logger):
        camera_info_dict = CameraInfoLoader.load_use_camera_info(camera_info_list_file, camera_setup_list_file,
                                                                 [camera_id], True, logger)
        if camera_id in camera_info_dict:
            return camera_info_dict[camera_id]
        return None

    @staticmethod
    def load_use_camera_info(camera_info_list_file, camera_setup_list_file, camera_id_url_list, is_id_list, logger):
        info_dict = dict()
        try:
            info_df = pd.read_csv(camera_info_list_file).values
            for info in info_df:
                info_dict[info[0]] = info[2:]

            if not camera_id_url_list:  # ImgAVR, WebAVR에서 사용
                camera_id_url_list = [item[0] for item in info_df]
                is_id_list = True

        except Exception as e:
            logger.error(" failed to open {} : {}".format(camera_info_list_file, e))

        setup_dict = dict()
        try:
            setup_df = pd.read_csv(camera_setup_list_file).values
            for setup in setup_df:
                setup_dict[setup[0]] = setup[1:]
        except Exception as e:
            logger.error(" failed to open {} : {}".format(camera_setup_list_file, e))

        camera_info_dict = dict()

        for camera_id_url in camera_id_url_list:
            if not is_id_list:
                camera_id = extract_camera_id(camera_id_url)
            else:
                camera_id = camera_id_url

            if camera_id in info_dict.keys():
                rtsp_url = info_dict[camera_id][0] if is_id_list else camera_id_url
                start_lane = info_dict[camera_id][1]
                lane_count = info_dict[camera_id][2]
                veh_dir_info = info_dict[camera_id][3]
                veh_side_info = info_dict[camera_id][4]
            else:
                if not is_id_list:
                    rtsp_url = camera_id_url
                    start_lane = 1
                    lane_count = 1
                    veh_dir_info = 0
                    veh_side_info = 1
                else:
                    continue

            if camera_id in setup_dict.keys():
                roi = setup_dict[camera_id][0]
                fps = int(setup_dict[camera_id][1])
            else:
                roi = "0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9"
                fps = 30

            roi = np.array([float(x) for x in roi.replace('"', '').split(',')]).reshape((4, 2)).tolist()

            camera_info_dict[camera_id] = CameraInfo(camera_id=camera_id, rtsp_url=rtsp_url, fps=fps, roi=roi,
                                                     start_lane=start_lane, lane_count=lane_count,
                                                     veh_dir_info=veh_dir_info, veh_side_info=veh_side_info)

        return camera_info_dict
