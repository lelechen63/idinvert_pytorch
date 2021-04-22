import os
import torch as th
import cv2
import numpy as np

import sys
sys.path.insert( 0, os.path.abspath( '..' ) )
from src.videoDrive.core.video_code_optim import VideoDataset, ProjectionLayer, load_cam_conf
from src.videoDrive.texUnwarp import texUnwarp

#add steve's utils
sys.path.append( '/mnt/captures/stephenlombardi/pyutils/' )
import pyutils


# -------------------- Output objs --------------------
def output_objs( video_dataset, coords, vert_idxs, coord_idxs, save_folder ):
    save_obj_folder = save_folder + '/mesh_no_rigid'
    if not os.path.exists( save_obj_folder ):
        os.makedirs( save_obj_folder )

    video_dataset.read_img = False
    video_dataset.read_tex = False
    video_dataset.read_lands = False

    frm_num = len( video_dataset )
    for idx in range( frm_num ):
        frm = video_dataset[ idx ]
        frm_id = frm[ 'frm_id' ]
        mesh_pt_xyzs = frm[ 'mesh_pt_xyzs']

        save_obj_path = save_obj_folder + '/%06d.obj' % frm_id
        pyutils.write_obj( save_obj_path, mesh_pt_xyzs, coords, vert_idxs, coord_idxs )

        print( 'Finish frame %d/%d' % ( idx + 1, frm_num ) )


# -------------------- Output images --------------------
def output_img( video_dataset, save_folder ):
    # Initialize projection layer
    proj_layer = ProjectionLayer()

    # Start to output
    frm_num = len( video_dataset )
    view_num = len( video_dataset.cam_names )
    video_dataset.read_img = True

    for frm_idx in range( frm_num ):
        frm = video_dataset[ frm_idx ]
        imgs = frm[ 'imgs' ]
        ext_mat = frm[ 'ext_mat' ]
        pt_xyzs = frm[ 'mesh_pt_xyzs' ]
        frm_id = frm[ 'frm_id' ]

        for view_id in range( view_num ):
            img = imgs[ view_id ].copy()

            view_save_folder = save_folder + '/images/' + video_dataset.cam_names[ view_id ]
            if not os.path.exists( view_save_folder ):
                os.makedirs( view_save_folder )

            save_img_np_path = view_save_folder + '/%06d' % frm_id
            np.save( save_img_np_path, img )

        print( 'Output image for frame %d/%d' %( frm_idx + 1, frm_num ) )


# -------------------- Output textures --------------------
def output_tex( video_dataset, coords, vert_idxs, coord_idxs, cam_ext_mats, cam_int_mats, save_folder ):
    # Parameters
    tex_width, tex_height = 1024, 1024

    # Initialize projection layer
    proj_layer = ProjectionLayer()

    # Build the index map
    idx_map_vec = texUnwarp.calcIndexMap( coords.transpose(), coord_idxs.transpose(), tex_width, tex_height )
    idx_map_vec = np.array( idx_map_vec, dtype = np.float32 )
    print( 'idx_map_vec', idx_map_vec.shape )

    # Start to output
    frm_num = len( video_dataset )
    view_num = len( video_dataset.cam_names )
    video_dataset.read_img_np = True

    for frm_idx in range( frm_num ):
        frm = video_dataset[ frm_idx ]
        imgs = frm[ 'imgs' ]
        ext_mat = frm[ 'ext_mat' ]
        pt_xyzs = frm[ 'mesh_pt_xyzs' ]
        frm_id = frm[ 'frm_id' ]

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
            img = imgs[ view_id ].copy()
            img_width, img_height = img.shape[ 1 ], img.shape[ 0 ]

            pt_uvws = pt_uvws_np[ view_id ]
            tex = texUnwarp.unwarpTex( img.reshape( -1 ), img_width, img_height,
                                       pt_uvws.transpose(), vert_idxs.transpose(),
                                       idx_map_vec, tex_width, tex_height )
            tex = np.array( tex, dtype = np.uint8 ).reshape( tex_height, tex_width, -1 )
            tex = cv2.flip( tex, 0 )

            save_tex_folder = save_folder + '/texes/' + video_dataset.cam_names[ view_id ]
            if not os.path.exists( save_tex_folder ):
                os.makedirs( save_tex_folder )
            save_tex_np_path = save_tex_folder + '/%06d' % frm_id
            np.save( save_tex_np_path, tex )

            #save_tex_path = save_tex_folder + '/%06d.png' % frm_id
            #cv2.imwrite( save_tex_path, cv2.cvtColor( tex, cv2.COLOR_RGB2BGR ) )

        print( 'Output texture for frame %d/%d' % ( frm_idx + 1, frm_num ) )


# -------------------- Normalize texture --------------------
def calc_tex_mean_var( video_dataset, save_folder ):
    video_dataset.read_img_np = False
    video_dataset.read_tex_np = True
    video_dataset.read_lands = False

    frm_num = len( video_dataset )
    view_num = len( video_dataset.cam_names )

    # Calculate the mean
    for frm_id in range( frm_num ):
        frm = video_dataset[ frm_id ]
        texes = frm[ 'texes' ]
        for view_id in range( view_num ):
            tex = texes[ view_id ][ : , : , : 3 ]

            if frm_id == 0 and view_id == 0:
                tot_tex = tex.astype( np.float32 )
                add_cnt = 1
            else:
                tot_tex += tex.astype( np.float32 )
                add_cnt += 1
        print( 'Sum up texture for frame %d/%d' % ( frm_id + 1, frm_num ) )

    mean_tex = tot_tex / float( add_cnt )
    mean_tex = mean_tex.clip( 0, 255.0 )
    mean_tex = cv2.cvtColor( mean_tex, cv2.COLOR_RGB2BGR )

    save_avrg_tex_path = save_folder + '/tex_mean.png'
    cv2.imwrite( save_avrg_tex_path, mean_tex )

    # Calculate the variance
    for frm_id in range( frm_num ):
        frm = video_dataset[ frm_id ]
        texes = frm[ 'texes' ]

        for view_id in range( view_num ):
            tex = texes[ view_id ][ : , : , : 3 ]
            tex_nor = tex - mean_tex
            var = ( tex_nor ** 2 ).mean()
            if frm_id == 0 and view_id == 0:
                tot_var = var
                add_cnt = 1
            else:
                tot_var += var
                add_cnt += 1
        print( 'Sum up variance for frame %d/%d' % ( frm_id + 1, frm_num ) )

    tex_std = np.sqrt( tot_var / float( add_cnt ) )
    save_tex_std_path = save_folder + '/tex_std.txt'
    with open( save_tex_std_path, 'w' ) as fp:
        fp.write( '%f' % tex_std )


# -------------------- Output position maps --------------------
def output_pos_maps( video_dataset, coords, vert_idxs, coord_idxs, save_folder ):
    # Parameters
    tex_width, tex_height = 1024, 1024

    # Initialize projection layer
    proj_layer = ProjectionLayer()

    # Build the index map
    idx_map_vec = texUnwarp.calcIndexMap( coords.transpose(), coord_idxs.transpose(), tex_width, tex_height )
    idx_map_vec = np.array( idx_map_vec, dtype = np.float32 )
    print( 'idx_map_vec', idx_map_vec.shape )

    # Start to output
    frm_num = len( video_dataset )
    video_dataset.read_img_np = False
    video_dataset.read_tex_np = False
    video_dataset.read_pos_map = False
    video_dataset.read_lands = False

    for frm_idx in range( frm_num ):
        frm = video_dataset[ frm_idx ]
        pt_xyzs = frm[ 'mesh_pt_xyzs' ]
        frm_id = frm[ 'frm_id' ]

        pos_map = texUnwarp.unwarpPos( pt_xyzs.transpose(), vert_idxs.transpose(),
                                       idx_map_vec, tex_width, tex_height )
        pos_map = np.array( pos_map, dtype = np.float32 ).reshape( tex_height, tex_width, -1 )
        pos_map = cv2.flip( pos_map, 0 )

        save_pos_map_folder = save_folder + '/pos_maps'
        if not os.path.exists( save_pos_map_folder ):
            os.makedirs( save_pos_map_folder )

        save_pos_map_path = save_pos_map_folder + '/%06d' % frm_id
        np.save( save_pos_map_path, pos_map )

        print( 'Finish frame %d/%d' % (frm_idx + 1, frm_num) )


# -------------------- Normalize position maps --------------------
def calc_pos_map_mean_var( video_dataset, save_folder ):
    video_dataset.read_img_np = False
    video_dataset.read_tex_np = False
    video_dataset.read_pos_map = True
    video_dataset.read_lands = False

    frm_num = len( video_dataset )
    for frm_idx in range( frm_num ):
        frm = video_dataset[ frm_idx ]
        pos_map = frm[ 'pos_map' ]

        if frm_idx == 0:
            tot_pos_map = pos_map
            add_cnt = 1
        else:
            tot_pos_map += pos_map
            add_cnt += 1
        print( 'Sum up pos_map for frame %d/%d' % ( frm_idx + 1, frm_num ) )

    pos_map_mean = tot_pos_map / float( add_cnt )

    save_pos_map_mean_path = save_folder + '/pos_map_mean.png'
    cv2.imwrite( save_pos_map_mean_path, pos_map_mean )
    save_pos_map_mean_path = save_folder + '/pos_map_mean.npy'
    np.save( save_pos_map_mean_path, pos_map_mean )

    tot_var = 0.0
    add_cnt = 0
    for frm_idx in range( frm_num ):
        frm = video_dataset[ frm_idx ]
        pos_map = frm[ 'pos_map' ]

        pos_map_nor = pos_map - pos_map_mean
        var = ( pos_map_nor ** 2 ).mean()

        tot_var += var
        add_cnt += 1
        print( 'Sum up variance for frame %d/%d' % ( frm_idx + 1, frm_num ) )

    pos_map_std = np.sqrt( tot_var / float( add_cnt ) )
    save_pos_map_std_path = save_folder + '/pos_map_std.txt'
    with open( save_pos_map_std_path, 'w' ) as fp:
        fp.write( '%f' % pos_map_std )


# -------------------- Main --------------------
if __name__ == '__main__':
    # Load video dataset
    data_folder = '/mnt/home/chen/camAvatarDrive/data/Chen/adhoc'
    track3D_pkl_path = data_folder + '/pairs_stage2/view_4/M1_Conversation_track3D.pkl'
    pt_xyzs_pkl_path = data_folder + '/pairs_stage2/view_4/M1_Conversation_pt_xyzs.pkl'
    save_folder = data_folder + '/input/M1_Conversation/visibility'

    video_dataset = VideoDataset()
    video_dataset.read_track3D_pkl( track3D_pkl_path )
    video_dataset.read_pt_xyzs_pkl( pt_xyzs_pkl_path )
    video_dataset.img_base_path = '/mnt/home/chen/camAvatarDrive/data/Chen/adhoc/input'
    video_dataset.lands_base_path = '/mnt/home/chen/camAvatarDrive/data/Chen/adhoc/input'
    video_dataset.tex_base_path = '/mnt/home/chen/camAvatarDrive/data/Chen/adhoc/input'
    video_dataset.pos_map_base_path = '/mnt/home/chen/camAvatarDrive/data/Chen/adhoc/input'
    video_dataset.cam_names = [ 'cam400291', 'cam400289', 'cam400348', 'cam400356' ]

    # Load standard objs
    ref_obj_path = '/mnt/home/chen/plan_to_move/camAvatarDrive/data/Common/rosettaFaceMesh.obj'
    verts, coords, vert_idxs, coord_idxs = pyutils.load_obj( ref_obj_path )
    verts = np.array( verts, dtype = np.float32 )
    coords = np.array( coords, dtype = np.float32 )
    vert_idxs = np.array( vert_idxs, dtype = np.float32 )
    coord_idxs = np.array( coord_idxs, dtype = np.float32 )

    # Load cameras
    cam_conf_path = data_folder + '/../track3DMV/cameraConf.txt'
    cam_names, cam_int_mats, cam_ext_mats = load_cam_conf( cam_conf_path )

    # # Output objs
    # output_objs( video_dataset, coords, vert_idxs, coord_idxs, save_folder )

    # Output the images

    # Output the textures
    output_tex( video_dataset, coords, vert_idxs, coord_idxs, cam_ext_mats, cam_int_mats, save_folder )

    # Calculate texture mean and variance
    #calc_tex_mean_var( video_dataset, save_folder )

    # Output position map
    # output_pos_maps( video_dataset, coords, vert_idxs, coord_idxs, save_folder )

    # Calculate pos_map mean and variance
    # calc_pos_map_mean_var( video_dataset, save_folder )