import glob
import os

import json
import numpy as np
import trimesh
import imageio
import openmesh
import cv2
from tqdm import tqdm
import pickle
import time, threading

image_data_root = "/raid/celong/FaceScape/fsmview_images"
landmark_root = "/raid/celong/FaceScape/fsmview_landmarks"
mesh_root = "/raid/celong/FaceScape/textured_meshes"

expressions = {
    1: "1_neutral",
    2: "2_smile",
    3: "3_mouth_stretch",
    4: "4_anger",
    5: "5_jaw_left",
    6: "6_jaw_right",
    7: "7_jaw_forward",
    8: "8_mouth_left",
    9: "9_mouth_right",
    10: "10_dimpler",
    11: "11_chin_raiser",
    12: "12_lip_puckerer",
    13: "13_lip_funneler",
    14: "14_sadness",
    15: "15_lip_roll",
    16: "16_grin",
    17: "17_cheek_blowing",
    18: "18_eye_closed",
    19: "19_brow_raiser",
    20: "20_brow_lower"
}

lm_list_v10 = np.load("./predef/landmark_indices.npz")['v10']

def get_face_orientation(id_idx, exp_idx, cam_idx):
    x_dir = np.array([1,0,0]).reshape(3,1)
    y_dir = np.array([0,1,0]).reshape(3,1)
    z_dir = np.array([0,0,1]).reshape(3,1)

    with open("./predef/Rt_scale_dict.json", 'r') as f:
        Rt_scale_dict = json.load(f)
        scale = Rt_scale_dict['%d'%id_idx]['%d'%exp_idx][0]
        Rt_TU = np.array(Rt_scale_dict['%d'%id_idx]['%d'%exp_idx][1])

    mesh_path = f"{mesh_root}/{id_idx}/models_reg/{expressions[exp_idx]}.obj"
    if not os.path.exists(mesh_path):
        print(f"[WARN] {mesh_path} not exist!")
        exit(0)

    om_mesh = openmesh.read_trimesh(mesh_path)
    verts = np.array(om_mesh.points())
    if (verts.shape[0] == 0):
        print(f"[WARN] {mesh_path} is empty!")
        exit(0)

    verts = (Rt_TU[:3,:3].T @ (verts - Rt_TU[:3,3]).T).T
    verts = verts / scale
    x_dir = Rt_TU[:3,:3].T @ x_dir
    y_dir = Rt_TU[:3,:3].T @ y_dir
    z_dir = Rt_TU[:3,:3].T @ z_dir

    img_dir = f"{image_data_root}/{id_idx}/{expressions[exp_idx]}"

    with open(f"{img_dir}/params.json", 'r') as f:
        params = json.load(f)

    K = np.array(params['%d_K' % cam_idx])
    Rt = np.array(params['%d_Rt' % cam_idx])
    h_src = params['%d_height' % cam_idx]
    w_src = params['%d_width' % cam_idx]
    
    R = Rt[:3,:3]
    T = Rt[:3,3:]

    lmks = verts[lm_list_v10]
    pos = K @ (R @ lmks.T + T) # (3,68)
    x_dir = R @ x_dir
    y_dir = R @ y_dir
    z_dir = R @ z_dir

    x_dir = x_dir / np.linalg.norm(x_dir)
    y_dir = y_dir / np.linalg.norm(y_dir)
    z_dir = z_dir / np.linalg.norm(z_dir)

    x_c = np.array([1,0,0]).reshape(3,1)
    y_c = np.array([0,-1,0]).reshape(3,1)
    z_c = np.array([0,0,-1]).reshape(3,1)

    return np.arccos(x_dir.T.dot(x_c)).squeeze() * 180 / np.pi, np.arccos(y_dir.T.dot(y_c)).squeeze() * 180 / np.pi, np.arccos(z_dir.T.dot(z_c)).squeeze() * 180 / np.pi

def get_all_folder_example():
    pids = os.listdir(image_data_root)
    for pid in pids:
        img_folder =  os.path.join(image_data_root, pid, '1_neutral')
        print (img_folder)
        command = 'cp -r ' + img_folder + ' ./tmp/' + pid
        print (command)
        os.system(command)

def get_front_pickle():
    gg =  open("./predef/frontface_list.txt", 'r')
    front_list = {}
    while True:
        line = gg.readline()[:-1]
        if not line:
            break
        print(line)
        tmp = line.split(',')
        print (tmp)
        print(tmp[0], tmp[1])
        front_list[tmp[0] +'__' + tmp[1]] = tmp[2]
    with open('./predef/frontface_list.pkl', 'wb') as handle:
        pickle.dump(front_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# get_front_list()
def  get_front_list():
    angle_lists =  open("./predef/angle_list.txt", 'r')
    total_list = {}
    front_list = {}
    while True:
        line = angle_lists.readline()[:-1]
        if not line:
            break
        print(line)
        tmp = line.split(',')
        print (tmp)
        print(tmp[0], tmp[1])
        total_list[tmp[0] +'__' + tmp[1] + '__' + tmp[2]] = [float(tmp[3]),float(tmp[4]), float(tmp[5])]
    print (total_list)

    pids = os.listdir(image_data_root)
    pids.sort()
    for id_idx in pids:
        for exp_id in range(len(expressions)):
            angles = []
            exp_idx = exp_id + 1
            for cam_idx in range(len(os.listdir(os.path.join( image_data_root , id_idx, expressions[exp_idx]))) -1):
                name_key = str(id_idx) +'__' + expressions[exp_idx] +'__' + str(cam_idx)
                if name_key in total_list.keys():
                    angles.append([ 10 * total_list[name_key][0] ,total_list[name_key][1],total_list[name_key][2]] )
            if len(angles) == 0:
                continue
            angles = np.array(angles)
            print (angles.shape)
            angle_sum = angles.sum(1)
            small_index = angle_sum.argsort()[0]
            front_list[str(id_idx) +'__' + expressions[exp_idx]] = [small_index]
    print (front_list)
    with open('./predef/frontface_list.pkl', 'wb') as handle:
        pickle.dump(front_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
def get_valid_pickle():
    angle_lists =  open("./predef/angle_list.txt", 'r')
    gg =  open("./predef/validface_list.txt", 'r')
    front_list = {}
    total_list = {}
    while True:
        line = angle_lists.readline()[:-1]
        if not line:
            break
        tmp = line.split(',')
        total_list[tmp[0] +'__' + tmp[1] + '__' + tmp[2]] = [float(tmp[3]),float(tmp[4]), float(tmp[5])]
    print (len(total_list))

    pids = os.listdir(image_data_root)
    pids.sort()
    for id_idx in pids:
        for exp_id in range(len(expressions)):
            angles = []
            exp_idx = exp_id + 1
            for cam_idx in range(len(os.listdir(os.path.join( image_data_root , id_idx, expressions[exp_idx]))) -1):
                name_key = str(id_idx) +'__' + expressions[exp_idx] +'__' + str(cam_idx)
                if name_key in total_list.keys():
                    # if 
                    angles.append([ total_list[name_key][0] ,total_list[name_key][1],total_list[name_key][2]] )

    while True:
        line = gg.readline()[:-1]
        if not line:
            break
        print(line)
        tmp = line.split(',')
        print (tmp)
        print(tmp[0], tmp[1])
        if tmp[0] + '__' + tmp[1] not in front_list.keys():
            front_list[tmp[0] + '__' + tmp[1] ] = [tmp[2]]
        else:
            front_list[tmp[0] + '__' + tmp[1] ].append(tmp[2])
    with open('./predef/validface_list.pkl', 'wb') as handle:
        pickle.dump(front_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_angle_batch(pid_b, i):
    angle_lists =  open("./predef/tmmp/angle_list_%d.txt"%i, 'w')
    for id_idx in pid_b:
        for exp_id in range(len(expressions)):
            angles = []
            exp_idx = exp_id + 1        
            for cam_idx in range(len(os.listdir(os.path.join( image_data_root , id_idx, expressions[exp_idx]))) -1):
                angle_x, angle_y, angle_z = get_face_orientation(int(id_idx), exp_idx, cam_idx)
                angle_lists.write(id_idx +',' + str(expressions[exp_idx]) + ',' + str(cam_idx) + ','  +  str(angle_x) + ','  +  str(angle_y)+ ','  +  str(angle_z) + '\n')
                print (id_idx +',' + str(expressions[exp_idx]) + ',' + str(cam_idx) + ','  +  str(angle_x) + ','  +  str(angle_y)+ ','  +  str(angle_z))
                print (i)


def  get_angle_list():
  
    pids = os.listdir(image_data_root)
    pids.sort()
    # gg =  open("./predef/validface_list.txt", 'w')
    N = 50
    batch = int(len(pids) /N)
    threads = []

    for i in range (N):
        angle_lists =  open("./predef/tmmp/angle_list_%d.txt"%i, 'w')
        threading.Thread(target = get_angle_batch, args = (pids[batch * i: batch *(i+1)], i)).start()
get_angle_list()
# get_valid_pickle()

# get_front_list()