#!/bin/bash

# 清空nvidia gpu 显存
# apt-get install psmisc 安装fuser

fuser /dev/nvidia* | awk '{for(i=1;i<NF;i++) print "kill -9 " $i;}' | sh