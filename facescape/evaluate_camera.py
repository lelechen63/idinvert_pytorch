import json
import numpy as np
import trimesh
import imageio
import cv2
if __name__ == '__main__':
    with open("/raid/celong/FaceScape/fsmview_images/123/13_lip_funneler/params.json", 'r') as f:
        params = json.load(f)
    
    test_num = 11
    scale =1.0

    K = np.array(params['%d_K' % test_num])
    Rt = np.array(params['%d_Rt' % test_num])
    h_src = params['%d_height' % test_num]
    w_src = params['%d_width' % test_num]

    # scale h and w
    h, w = int(h_src * scale), int(w_src * scale)

    dist = np.array(params['%d_distortion' % test_num], dtype = np.float)

    # read image
    src_img = cv2.imread("/raid/celong/FaceScape/fsmview_images/123/13_lip_funneler/%d.jpg" % test_num)
    src_img = cv2.resize(src_img, (w, h))
    # undistort image
    undist_img = cv2.undistort(src_img, K, dist)

    h_src = params['%d_height' % test_num]
    w_src = params['%d_width' % test_num]

    R_cv2gl = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
    # Rt = R_cv2gl.dot(Rt)

    mesh_dirname = "/raid/celong/FaceScape/fsmview_shapes/123/13_lip_funneler.ply"
    mesh = trimesh.load_mesh(mesh_dirname)
    verts = np.array(mesh.vertices)
    print (verts.shape)
    
    verts_color = np.zeros(verts.shape)
    colored_verts = np.concatenate(verts, verts_color,axis = 1)
    print (colored_verts.shape)

    R = Rt[:3,:3]
    T = Rt[:3,3:]

    pos = K @ (R @ verts.T + T)
    coord = pos[:2,:] / pos[2,:]
    
    coord = coord.astype(int)
    print (coord.shape)
    coord[0, :] = np.clip(coord[0, :], 0, w_src - 1)
    coord[1, :] = np.clip(coord[1, :], 0, h_src - 1)
    print (coord.shape,'1')
    coord = coord[::-1,:]
    print (coord.shape,'2')
    img = np.zeros((h_src, w_src))
    print (img.shape)
    print (coord.shape, "!!!")
    img[tuple(coord)] = 1.0
    imageio.imsave(f"mask_{test_num}.jpg", img)

    print (undist_img.shape,'@@')