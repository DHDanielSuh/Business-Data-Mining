#!/usr/bin/env python
# -*- coding:utf-8 -*-
# from threading import Thread
from Common.AVRlib import *
import threading
import os
import shutil
import cv2
import sys
import argparse
import datetime
import time
import multiprocessing
from ffmpy import FFmpeg

__author__ = "Kim, Sungsoo"
__copyright__ = "Copyright 2019, The AI Vehicle Recognition Project"
__credits__ = ["Kim, Sungsoo", "Paek, Hoon", ]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = ["Kim, Sungsoo", "Paek, Hoon", ]
__email__ = ["onair0817@mindslab.ai", "hoon.paek@mindslab.ai"]
__status__ = "Development"  # Development / Test / Release.


CODEC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# CODEC = cv2.VideoWriter_fourcc('X','V','I','D')

VIDEO_EXT = '.avi'
# IMAGE_EXT = '.png'
TEST_THREAD = False
CHECK_FRAME_RATE_ = False
CAPTURE_INTERVAL_ADJUSTMENT_RATIO = 0.7

DEBUG_LOG_RING_BUFFER = False
#DEBUG_LOG_RING_BUFFER = True

_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


class RingBuffer:
    class FrameInfo:
        def __init__(self, frame):
            self.time = datetime.datetime.now()
            self.frame = frame

    def __init__(self, size, logger=None):
        self.size = size
        self.buffer = [None, ] * self.size
        self.write_ptr = 0
        self.read_ptr = 0
        self.logger = logger
        self.lock = threading.Condition()

    def debug_log(self,  msg):
        if DEBUG_LOG_RING_BUFFER and self.logger is not None:
            self.logger.info(" RingBuffer.{}. w{:d}, r{:d}".format(msg, self.write_ptr, self.read_ptr))

    def write(self, frame):
        # time.sleep(5)  #set writer slower

        self.debug_log("write:Entering")
        with self.lock:
            self.debug_log("write:Entered")

            self.buffer[self.write_ptr] = RingBuffer.FrameInfo(frame)
            """
            start 변수를 두지 않아도 되지만, 마지막 하나는 사용하지 못한다.
            즉, rtsp가 ㅑmgExt 보다 빠를 경우 size-1 만큼만 사용하게 됨
            """
            self.write_ptr = (self.write_ptr + 1) % self.size
            if self.write_ptr == self.read_ptr:
                if self.logger is not None:
                    self.logger.info(
                        " RingBuffer.write. No empty buffer. inc r. w{:d}, r{:d}".format(self.write_ptr, self.read_ptr))
                self.read_ptr = (self.read_ptr + 1) % self.size

            self.lock.notify()

        self.debug_log("write:Leaved")

    def read(self):
        # time.sleep(5)  #set reader slower

        self.debug_log("read:Entering")

        with self.lock:
            self.debug_log("read:Entered")

            # write_ptr 전까지는 read 가능. 즉, write_ptr 이 5이면 4까지는 write 했다는 것임
            if self.read_ptr == self.write_ptr:
                self.debug_log("read:Wait")

                self.lock.wait(1)

                self.debug_log("read:Waked")
                # wait이 끝나면, 다시 read 가 호출되어 처리하도록 한다.
                return None

            assert(self.buffer[self.read_ptr] is not None)
            assert(self.buffer[self.read_ptr].frame is not None)

            frame = self.buffer[self.read_ptr].frame

            self.read_ptr = (self.read_ptr + 1) % self.size

            self.debug_log("read:Leaved")
            return frame


class RTSPClient:
    def __init__(self, ini, rtsp_url, logger=None):
        self.ini = ini
        self.rtsp_url = rtsp_url
        self.save_time = None

        self.retry_interval = None
        self.capture_interval = None

        self.ring_buf = None

        self.logger = logger

        self.out_vid_fname = None
        self.output_vid = None

        self.capture = None
        self.status = None
        self.frame = None

        self.ref_sec = -1
        self.frame_cnt = 0
        self.thread_run_ = True

        self.init_ini(ini['RTSP_CLIENT'])

    def init_ini(self, ini):
        self.save_time = int(ini['save_time'])

        self.retry_interval = int(ini['retry_interval'])
        self.capture_interval = float(ini['capture_interval'])
        self.capture_interval *= CAPTURE_INTERVAL_ADJUSTMENT_RATIO

        self.ring_buf = RingBuffer(int(ini['ring_buffer_size']), self.logger)

    @staticmethod
    def make_output_path(folder, file_name, logger):
        """
        cur_name = file_name
        for idx in range(5):
            out_path = os.path.join(folder, cur_name) + VIDEO_EXT
            if not os.path.exists(out_path):
                return out_path

            logger.info("Same file name exist. : {}".format(out_path))
            cur_name = file_name + idx
        return None
        """
        return os.path.join(folder, file_name) + VIDEO_EXT

    def initialize(self, folder):
        self.capture = cv2.VideoCapture(self.rtsp_url)

        if not self.capture.isOpened():
            self.logger.error(" @ Error: OpenCV VideoCapture is not opened : {}".format(self.rtsp_url))
            return False

        frame_width = int(self.capture.get(3))
        frame_height = int(self.capture.get(4))
        fps = self.capture.get(5)
        self.logger.info(" # {} : ({:d} x {:d}) @ {:5.2f} Hz".
                         format(self.rtsp_url, frame_width, frame_height, fps))

        camera_id = self.rtsp_url.rsplit('/', 1)[1] + '_'

        self.out_vid_fname = RTSPClient.make_output_path(folder, camera_id, self.logger)
        if self.out_vid_fname is None:
            self.logger.error('failed create file name of video')
            return False

        self.output_vid = cv2.VideoWriter(self.out_vid_fname, CODEC, fps, (frame_width, frame_height))
        return True

    def capturing_image_thread(self):
        self.logger.info(" Capture Thread.start : {}".format(self.rtsp_url))
        while self.thread_run_:
            self.check_frame_rate(flag=False, prefix="local")
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    self.ring_buf.write(frame)
                else:
                    self.logger.info(" Capture Thread. Fail to Read : {}".format(self.rtsp_url))
            else:
                self.logger.error(" Capture Thread. capture is NOT opened! : {}".format(self.rtsp_url))
                break
            time.sleep(self.capture_interval)
        self.logger.info(" Capture Thread.stop : {}".format(self.rtsp_url))

    def check_frame_rate(self, flag=True, prefix=""):
        if flag:
            crt_sec = datetime.datetime.now().second
            if crt_sec == self.ref_sec:
                self.frame_cnt += 1
            else:
                # self.logger.info(" # {} # Video real FPS: {:2d}".format(prefix, self.frame_cnt))
                self.frame_cnt = 0
                self.ref_sec = crt_sec

    def show_frame(self, frame):
        cv2.imshow('RTSP Client {}'.format(__version__), frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.logger.info(" # Stop streaming, {}".format(self.rtsp_url))
            self.capture.release()
            if self.output_vid is not None:
                self.output_vid.release()
                self.logger.info(" # {} is saved as {}".format(self.rtsp_url, self.out_vid_fname))
            cv2.destroyAllWindows()
            sys.exit(1)

    def save_frame(self, frame):
        if self.output_vid:
            self.output_vid.write(frame)

    def run(self, show_video):
        thread = threading.Thread(target=self.capturing_image_thread, args=())
        thread.daemon = True
        thread.start()

        time.sleep(1)

        start_time = time.time()

        while True:
            frame = self.ring_buf.read()
            if frame is not None:
                try:
                    self.check_frame_rate(flag=True, prefix="main")

                    if show_video:
                        self.show_frame(frame)
                    if self.output_vid:
                        self.save_frame(frame)

                except Exception as e:
                    self.logger.error(" RTSPClient:{}. @ Exception in the loop : {}".format(self.rtsp_url, e))
                    break

            elapsed = time.time() - start_time
            if elapsed > self.save_time:
                break

        self.thread_run_ = False
        time.sleep(1)
        thread.join()
        del thread

        self.capture.release()

        if self.output_vid:
            self.output_vid.release()

        self.transform_video()

    def transform_video(self):
        file_path = self.out_vid_fname

        start_time = time.time()

        trans_path = file_path.rsplit('.', 1)[0] + '.mp4'
        self.logger.info(' transform_video.start transform {} to mpeg({})'.format(file_path, trans_path))

        #os.remove(trans_path)
        # ff = FFmpeg(inputs={file_path: None}, outputs={trans_path: '-y -loglevel quiet -stats -c:v libx264'})
        ff = FFmpeg(inputs={file_path: None}, outputs={trans_path: '-y -c:v libx264'})
        ff.run()
        self.logger.info(' transform_video.end transform {}:{:f}mb to {}:{:f}mb'.format(
            file_path, os.path.getsize(file_path)/(1024*1024), trans_path, os.path.getsize(trans_path)/(1024*1024)))
        os.remove(file_path)

        elapsed = time.time() - start_time
        self.logger.info(' transform_video. total elapse {:.4f}'.format(elapsed))


class MultiProcessing(multiprocessing.Process):
    def __init__(self, ini, rtsp_url, show_video, folder, logger):
        multiprocessing.Process.__init__(self)
        self.ini = ini
        self.rtsp_url = rtsp_url
        self.show_video = show_video
        self.folder = folder
        self.logger = logger

    def run(self):
        self.logger.info(' # Start Process : {}'.format(self.rtsp_url))

        try:
            client = RTSPClient(ini=self.ini, logger=self.logger, rtsp_url=self.rtsp_url)
            if not client.initialize(self.folder):
                self.logger.error(' # failure of initialization. Process({}) of Multi process'.format(self.rtsp_url))
                return False
            client.run(self.show_video)
        except Exception as e:
            self.logger.info(' # Process {}. exception:{}'.format(self.rtsp_url, e))
            return

        self.logger.info(' # End Process : {}'.format(self.rtsp_url))


def main(args):
    #folder = os.path.join(args.out_path, utils.get_datetime()[:-3].replace(":", "-"))
    folder = args.out_path
    if os.path.exists(folder):
        shutil.rmtree(folder)

    if not os.path.exists(folder):
        os.makedirs(folder)
    ini = utils.get_ini_parameters(args.ini_fname)
    logger = utils.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_,
                                         console_=True)
    if args.out_path is not None:
        utils.folder_exists(args.out_path, exit_=False, create_=True, print_=False)

    if args.url_list is not None:
        url_list_file = args.url_list
    elif 'url_list_file' in ini['RTSP_CLIENT']:
        url_list_file = ini['RTSP_CLIENT']['url_list_file']
    else:
        logger.error('no url list file')
        return

    url_batch_size = int(ini['RTSP_CLIENT']['url_batch_size'])
    if url_batch_size < 1:
        url_batch_size = 1

    save_time = int(ini['RTSP_CLIENT']['save_time'])
    start_time = time.time()

    logger.info("")
    logger.info(" # START {}. list:{}, batch:{:d}, save time:{:d}sec".format(_this_basename_,
                                                                url_list_file, url_batch_size, save_time))

    rtsp_url_list = None
    with open(url_list_file, 'r') as f:
        rtsp_url_list = f.read().splitlines()
        logger.info("total {:d} in {}".format(len(rtsp_url_list), url_list_file))

        fail_list = []
        for start in range(0, len(rtsp_url_list), url_batch_size):
            thread_list = []
            for idx in range(start, start+url_batch_size):
                if idx >= len(rtsp_url_list):
                    break

                thread = MultiProcessing(ini, rtsp_url_list[idx], args.show_video_, folder, logger)
                """
                if not thread.initialize():
                    fail_list.append(rtsp_url_list[idx])
                    continue
                """

                thread.daemon = True
                thread_list.append(thread)

            logger.info(" # Starting RTSP threads. {:d}~. {:d} threads...".format(
                                                                                start, start+len(thread_list)))
            for thread in thread_list:
                thread.start()

            for thread in thread_list:
                thread.join()
                """
                if thread.exitcode != 0:
                    fail_list.append(thread.rtsp_url)
                """
            logger.info(" # Terminated RTSP threads...")

    elapsed = time.time() - start_time
    logger.info(" # End {}. total elapse {:.4f}".format(_this_basename_, elapsed))
    if rtsp_url_list is None:
        logger.info(" fail read list {}".format(url_list_file))
    """
    else:
        if len(fail_list) > 0:
            logger.info(" fail list :\n{}".format(fail_list))
        else:
            logger.info(" total list[{:d}] success".format(len(rtsp_url_list)))
    """


INI_FNAME = _this_basename_ + '.ini'
OUT_PATH = "RTSPTestOutput"


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--url_list", default=None, help="ini filename")
    parser.add_argument("--ini_fname", default=INI_FNAME, help="ini filename")
    parser.add_argument("--out_path", default=OUT_PATH, help="Output path")
    parser.add_argument("--show_video_", default=False, action='store_true', help="Show video")
    parser.add_argument("--logging_", default=True, action='store_true', help="Logging flag")
    parser.add_argument("--console_logging_", default=True, action='store_true', help="Console logging flag")

    args = parser.parse_args(argv)
    args.out_path = utils.unicode_normalize(args.out_path)

    return args


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
