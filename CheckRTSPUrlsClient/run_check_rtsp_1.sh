#!/usr/bin/env bash

AVR_DIR="/gtpa/ais/dev/AI-Vehicle-Recognition"

source ${HOME}/.bashrc
source activate avr

cd ${AVR_DIR}
source env_setup.sh

cd CheckRTSPUrlsClient

nohup python CheckRTSPUrlsClient.py --url_list rtsp_url1.txt &

