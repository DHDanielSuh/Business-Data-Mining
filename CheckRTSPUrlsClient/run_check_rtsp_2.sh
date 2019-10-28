#!/usr/bin/env bash

AVR_DIR="/gtpa/ais/dev/AI_Vehicle_Recognition"

source ${HOME}/.bashrc
source activate avr

cd ${AVR_DIR}
cd ImgExtract

nohup python CheckRTSPUrlsClient.py --url_list rtsp_url2.txt &

