#! /usr/bin/env python
# -*- coding: utf-8 -*-

from Common.AVRlib import *
import os
import shutil


class MMapFileManager:
    def __init__(self):
        self.mmap_arr = None
        self.mmap_file_num = None
        self.mmap_idx = None
        self.mmap_fnames = None
        self.mmap_shape = None
        self.mmap_fname = None
        self.mmap_frame = None

    def init_mmap_files(self, mmap_ini, logger, prefix, mmap_shape):
        logger.info("init mmap files : {}. {}".format(prefix, mmap_shape))

        self.mmap_idx = 0
        self.mmap_file_num = int(mmap_ini['mmap_file_num'])
        mmap_file_dir = os.path.join(mmap_ini['mmap_file_dir'], prefix)
        # noinspection PyUnreachableCode
        if __debug__:
            mmap_file_dir += "_" + utils.get_datetime().replace(":","-")[:-7]
        mmap_file_prefix = mmap_ini['mmap_file_prefix']

        try:
            if utils.folder_exists(mmap_file_dir, exit_=False, create_=False, print_=False):
                shutil.rmtree(mmap_file_dir)
            os.makedirs(mmap_file_dir)
        except Exception as ex:
            logger.error(" @ Error: roi_mmap file dir handling failed : {}".format(ex))
            sys.exit(1)

        self.mmap_shape = mmap_shape

        dummy = np.zeros(shape=self.mmap_shape, dtype='uint8')

        self.mmap_fnames = []
        self.mmap_arr = []
        abs_path = os.path.abspath(mmap_file_dir)
        for i in range(self.mmap_file_num):
            self.mmap_fnames.append(os.path.join(abs_path, "{}.mmap_{:02d}.mmap".format(mmap_file_prefix, i)))
            mmap = np.memmap(self.mmap_fnames[i], dtype='uint8', mode='w+', shape=tuple(self.mmap_shape))
            mmap[:] = dummy
            mmap.flush()
            self.mmap_arr.append(mmap)

        logger.info(" # init_mmap_files mmap file dir : {}".format(mmap_file_dir))

    def check_init_mmap(self, mmap_ini, logger, prefix, mmap_shape):
        if self.mmap_arr and self.mmap_shape == mmap_shape:
            return

        self.init_mmap_files(mmap_ini, logger, prefix, mmap_shape)

    def write_mmap(self, frame):
        self.mmap_arr[self.mmap_idx][:] = frame[:]
        self.mmap_fname = self.mmap_fnames[self.mmap_idx]
        self.mmap_frame = self.mmap_arr[self.mmap_idx]
        self.mmap_idx = (self.mmap_idx + 1) % self.mmap_file_num

        return self.mmap_fname

    @staticmethod
    def read_mmap(fname, shape):
        return np.memmap(fname, dtype='uint8', mode='r', shape=tuple(shape))
