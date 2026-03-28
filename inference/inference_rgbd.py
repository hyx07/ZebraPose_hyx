from zebrapose import zebra
from yolo import yolo
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import json

obj_dict = {
    'coconut':(1, 0),#(model id, yolo id)
    'red_soap':(3, 1),
    'blue_soap_box':(6, 2),
    'orange_marker':(7 ,3),
    'banana':(12, 4),
    'canned_pepsi':(16, 5),
    'mango':(26, 6),
    'yibao':(32, 7),
    'red_opao':(34, 8),
    'red_whiteboard_pen':(35, 9)
    }

###
obj_name = 'coconut'

K = [616.58,0,323.103, 0,616.778,238.464, 0.0, 0.0, 1.0]
###

def prj_marker2img(point,R,T,K):
    point_cam = np.matmul(R, point) + T
    point_pix = np.matmul(K, point_cam)
    point_pix = point_pix/point_pix[2]
    return point_pix[:2].astype(int)

def load_dims(model_id):
    with open('./assets/models_info.json', 'r') as f:
        model_info = json.load(f)

    info = model_info[str(model_id)]
    dims = [info['size_x'], info['size_y'], info['size_z']]
    return dims

dims = load_dims(obj_dict[obj_name][0])
zebra = zebra(obj_name, obj_dict[obj_name][0], K)
yolo = yolo(obj_dict[obj_name][1])

pipeline = rs.pipeline()
config = rs.config()
align = rs.align(rs.stream.color)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

time_array = np.zeros(10)
time_index = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        begin = time.time()
        bboxes = yolo.predict(color_image)
        yolo_time = time.time() - begin
        print('yolo time:', yolo_time)

##        for bbox in bboxes:
##            bbox = bbox.astype(int)
##            x,y,w,h = bbox
##            cv2.rectangle(color_image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)),(0,255,0),1)

        r_list, t_list = [], []
        for bbox in bboxes:
            r_predict,t_predict,success = zebra.predict(color_image, bbox, debug=False)
            if success:
                r_list.append(r_predict)
                t_list.append(t_predict) 

        time_elapsed = time.time() - begin
        time_array[time_index] = time_elapsed
        time_index = (time_index+1)%10

        for i in range(len(r_list)):
            r_predict = r_list[i]
            t_predict = t_list[i]
            
            rot_mat = r_predict
            point_center = t_predict[:,0]
            K = zebra.cam_K

            point_object = np.array([dims[0]/2, dims[1]/2, dims[2]/2])
            point_img1 = prj_marker2img(point_object,rot_mat,point_center,K)

            point_object = np.array([-dims[0]/2, dims[1]/2, dims[2]/2])
            point_img2 = prj_marker2img(point_object,rot_mat,point_center,K)

            point_object = np.array([dims[0]/2, -dims[1]/2, dims[2]/2])
            point_img3 = prj_marker2img(point_object,rot_mat,point_center,K)

            point_object = np.array([-dims[0]/2, -dims[1]/2, dims[2]/2])
            point_img4 = prj_marker2img(point_object,rot_mat,point_center,K)

            point_object = np.array([dims[0]/2, dims[1]/2, -dims[2]/2])
            point_img5 = prj_marker2img(point_object,rot_mat,point_center,K)

            point_object = np.array([-dims[0]/2, dims[1]/2, -dims[2]/2])
            point_img6 = prj_marker2img(point_object,rot_mat,point_center,K)

            point_object = np.array([dims[0]/2, -dims[1]/2, -dims[2]/2])
            point_img7 = prj_marker2img(point_object,rot_mat,point_center,K)

            point_object = np.array([-dims[0]/2, -dims[1]/2, -dims[2]/2])
            point_img8 = prj_marker2img(point_object,rot_mat,point_center,K)  
            
            cv2.line(color_image, point_img5, point_img6, (255,0,0),1)
            cv2.line(color_image, point_img6, point_img8, (255,0,0),1)
            cv2.line(color_image, point_img8, point_img7, (255,0,0),1)
            cv2.line(color_image, point_img5, point_img7, (255,0,0),1)

            cv2.line(color_image, point_img1, point_img5, (0,255,0),1)
            cv2.line(color_image, point_img2, point_img6, (0,255,0),1)
            cv2.line(color_image, point_img3, point_img7, (0,255,255),1)
            cv2.line(color_image, point_img4, point_img8, (0,255,255),1)
            
            cv2.line(color_image, point_img1, point_img2, (0,0,255),1)
            cv2.line(color_image, point_img2, point_img4, (0,0,255),1)
            cv2.line(color_image, point_img4, point_img3, (0,0,255),1)
            cv2.line(color_image, point_img1, point_img3, (0,0,255),1)

        fps = 1/np.mean(time_array)
        cv2.putText(color_image, 'FPS:%.1f'%fps, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
       
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('image', color_image)
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        
finally:
    pipeline.stop()
