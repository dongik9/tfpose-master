from glob import glob
from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from rest_framework.views import APIView
import cv2

import argparse
import logging
import time
import math, datetime
import pygame
import pandas as pd
import keyboard
from pprint import pprint
import numpy as np
import sys
from sklearn.svm import SVC

from .openpose.tf_pose.estimator import TfPoseEstimator
from .openpose.tf_pose.networks import get_graph_path, model_wh
import os
from rest_framework.response import Response
from rest_framework.decorators import api_view

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


cred = credentials.Certificate("C:\\Users\\AI04\\Desktop\\tfpose-master\mysite\webcam\\airbag-34576-firebase-adminsdk-369sa-74b2c5b21d.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://airbag-34576-default-rtdb.firebaseio.com/'
    #'databaseURL' : '데이터 베이스 url'
})


# Create your views here.
def index(request):
    return render(request,'index.html')

# load model
model = 'mobilenet_thin'


def stream():
    # logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
    
    ##########################################################################################

    w, h = model_wh('0x0')
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # if(args.camera == '0'):
    #     file_write_name = 'camera_0'
    # else:
    #     #basename = os.path.basename(args.camera)
    #     # path = os.path.dirname(imgfile)
    #     file_write_name, _ = os.path.splitext(args.camera) 
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    count = 0
    y1 = [0,0]
    frame = 0
    dataset = []
    state = []

    # 모델 학습
    data = pd.read_csv('C:\\Users\\AI04\\Desktop\\tfpose-master\\mysite\\webcam\\posedata.csv')
    X, Y = data.iloc[:,:36], data['class']
    x = X.to_numpy()
    y = Y.to_numpy()
    svc_model = SVC(kernel='poly')
    svc_model.fit(x, y)
    


    while True:
        tt = "no"
        ret_val, image = cam.read()
        # ############### image = cv2.resize(image, dsize=(300, 300), interpolation=cv2.INTER_AREA)
        i =1
        count+=1
        if count % 11 == 0:
            continue
        # logger.debug('image process+')
        if not ret_val:
            break
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        # In humans total num of detected person in frame
        # logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        # logger.debug('show+')
        for human in humans:
            # we select one person from num of person
            for i in range(len(humans)):
                try:
                    # human.parts contains all the detected body parts
                    a = human.body_parts[0]  # human.body_parts[0] is for head point coordinates
                    x = a.x*image.shape[1]   # x coordinate relative to image 
                    y = a.y*image.shape[0]   # y coordinate relative to image
                    y1.append(y)   # store value of y coordinate in list to compare two frames

                    ## 실시간 좌표값 기반 모델 예측
                    x1 = [0 for i in range(0,36)]       # 실시간 좌표값
                    for j in range(0,34):
                        if human.body_parts[j].x != None:
                            x1[2*j] = human.body_parts[j].x
                            x1[2*j+1] = human.body_parts[j].y
                        result = svc_model.predict([x1])
                        if result == [0]:               # stand
                            tt = "stand"
                            state.append(0)
                            begin = time.time()
                            ref = db.reference() #db 위치 지정, 기본 가장 상단을 가르킴
                            ref.update({'situation' : tt}) #해당 변수가 없으면 생성한다.
                        elif result == [1]:             # sit
                            tt = "sit"
                            state.append(1)
                            begin = time.time()
                        elif result == [2]:             # lie
                            tt = "lie"
                            if len(state) != 0:
                                end = time.time()
                                if state[-1] == 0 or state[-1] == 1:   # 전 값이 서있는 상태 또는 앉아있는 상태
                                    if (end-begin) <= 2:
                                        print("end ", end)
                                        print("begin ", begin)
                                        print("fall이 출력되어야함")
                                        tt = "fall"
                                        print(tt)
                                        ref = db.reference() #db 위치 지정, 기본 가장 상단을 가르킴
                                        ref.update({'situation' : tt}) #해당 변수가 없으면 생성한다.


                                    state = []
                        elif result == [3]:             # normal
                            tt = "normal"
                            state = []
                except:
                    pass
    ###################################################################################################################
        # cv2.putText(image,
        #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        # fps_time = time.time()
        # 
        # 

        # cv2.putText(image, tt, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,0,0), 3)
        ####cv2.imshow('tf-pose-estimation result', image)

        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--image\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')  
               
        
        
        # if(frame == 0) and (args.save_video):   # It's use to intialize video writer ones
        #     out = cv2.VideoWriter(file_write_name+'_output.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        #             20,(image.shape[1],image.shape[0]))
        # out.write(image)
        if cv2.waitKey(1) == 27:
            break
        # logger.debug('finished+')

        # response_dict = {"situation": tt}
        
        # return Response(response_dict, status=201)

    cv2.destroyAllWindows()

def video_feed(request):
    return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=image')    