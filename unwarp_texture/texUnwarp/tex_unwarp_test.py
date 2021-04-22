import os
import numpy as np
import cv2
import torch as th

import sys
sys.path.insert( 0, os.path.abspath( '../' ) )
from src.videoDrive.core.video_code_optim import VideoDataset, ProjectionLayer
from src.videoDrive.core.video_code_optim import load_cam_conf
from src.videoDrive.texUnwarp import texUnwarp
from src.videoDrive.face3DTrack.render_layer import Rodrigues, invRodrigues

sys.path.append( '/mnt/captures/stephenlombardi/pyutils/' )
import pyutils

if __name__ == '__main__':
    # Parameters
    tex_width, tex_height = 1024, 1024

    # Load video dataset
    data_folder = '/mnt/home/chen/camAvatarDrive/data/Chen/adhoc'
    track3D_pkl_path = data_folder + '/pairs_stage2/view_4/M1_Conversation_track3D.pkl'
    pt_xyzs_pkl_path = data_folder + '/pairs_stage2/view_4/M1_Conversation_pt_xyzs.pkl'
    save_folder = data_folder + '/track3DMV/test'

    video_dataset = VideoDataset()
    video_dataset.read_track3D_pkl( track3D_pkl_path )
    video_dataset.read_pt_xyzs_pkl( pt_xyzs_pkl_path )
    video_dataset.img_base_path = '/mnt/captures/studies/projects/GHS/Mugsy/m--20190718--1323--3279951--pilot--telepresenceV1/processed_GHS_v1/adhoc_0.5x'
    video_dataset.cam_names = [ 'cam400291', 'cam400289', 'cam400348', 'cam400356' ]

    # Load standard objs
    ref_obj_path = '/mnt/home/chen/plan_to_move/camAvatarDrive/data/Common/rosettaFaceMesh.obj'
    verts, coords, vert_idxs, coord_idxs = pyutils.load_obj( ref_obj_path )
    verts = np.array( verts, dtype = np.float32 )
    coords = np.array( coords, dtype = np.float32 )
    vert_idxs = np.array( vert_idxs, dtype = np.float32 )
    coord_idxs = np.array( coord_idxs, dtype = np.int32 )

    # Calculate index map
    idx_map_vec = texUnwarp.calcIndexMap( coords.transpose(), coord_idxs.transpose(), tex_width, tex_height )
    idx_map_vec = np.array( idx_map_vec, dtype = np.float32 )
    print( 'idx_map_vec', idx_map_vec.shape )

    # Load cameras
    cam_conf_path = data_folder + '/../track3DMV/cameraConf.txt'
    cam_names, cam_int_mats, cam_ext_mats = load_cam_conf( cam_conf_path )

    # Test
    proj_layer = ProjectionLayer()

    frm_num = len( video_dataset )
    view_num = len( video_dataset.cam_names )

    video_dataset.read_img = True
    for frm_id in range( frm_num ):
        frm = video_dataset[ frm_id ]

        imgs = frm[ 'imgs' ]
        ext_mat = frm[ 'ext_mat' ]
        pt_xyzs = frm[ 'mesh_pt_xyzs' ]

        t_pt_xyzs = []
        t_rot_mat = []
        t_trans = []
        t_int_mat = []
        for view_id in range( view_num ):
            cam_ext_mat = cam_ext_mats[ view_id ]
            cam_int_mat = cam_int_mats[ view_id ]

            view_ext_mat = cam_ext_mat.dot( ext_mat )
            t_rot_mat.append( view_ext_mat[ : 3, : 3 ] )
            t_trans.append( view_ext_mat[ : 3, 3 ] )
            t_pt_xyzs.append( pt_xyzs )
            t_int_mat.append( cam_int_mat )
        t_rot_mat = th.tensor( t_rot_mat, dtype = th.float32 )
        t_trans = th.tensor( t_trans, dtype = th.float32 )
        t_pt_xyzs = th.tensor( t_pt_xyzs, dtype = th.float32 )
        t_int_mat = th.tensor( t_int_mat, dtype = th.float32 )

        t_pt_uvws = proj_layer( t_pt_xyzs, t_rot_mat, t_trans, t_int_mat )
        pt_uvws_np = t_pt_uvws.numpy()

        for view_id in range( view_num ):
            img = imgs[ view_id ]
            img = cv2.cvtColor( img, cv2.COLOR_RGB2BGR )
            img_width, img_height = img.shape[ 1 ], img.shape[ 0 ]

            pt_uvws = pt_uvws_np[ view_id ]
            tex = texUnwarp.unwarpTex( img.reshape( -1 ), img_width, img_height,
                                       pt_uvws.transpose(), vert_idxs.transpose(),
                                       idx_map_vec, tex_width, tex_height )
            tex = np.array( tex, dtype = np.uint8 ).reshape( tex_height, tex_width, -1 )
            tex = cv2.flip( tex, 0 )
            save_tex_path = save_folder + '/view%d_tex.png' % view_id
            cv2.imwrite( save_tex_path, tex )

            for pt_id in range( pt_uvws.shape[0] ):
                pt_uvw = pt_uvws[ pt_id ]
                cv2.circle( img, ( int( pt_uvw[ 0 ] + 0.5 ), int( pt_uvw[ 1 ] + 0.5 ) ), 1,
                            ( 0, 0, 255 ), -1 )

            img = cv2.flip( img, 0 )
            save_img_path = save_folder + '/view%d.png' % view_id
            cv2.imwrite( save_img_path, img )

        break
