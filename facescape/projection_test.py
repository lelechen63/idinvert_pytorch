import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import sys, cv2, json
import numpy as np
import renderer

def projection_test(test_num, scale=1.0):

    # read params
    with open("/raid/celong/FaceScape/fsmview_images/123/13_lip_funneler/params.json", 'r') as f:
        params = json.load(f)

    # extract KRt dist
    K = np.array(params['%d_K' % test_num])
    Rt = np.array(params['%d_Rt' % test_num])
    dist = np.array(params['%d_distortion' % test_num], dtype = np.float)
    h_src = params['%d_height' % test_num]
    w_src = params['%d_width' % test_num]

    # scale h and w
    h, w = int(h_src * scale), int(w_src * scale)
    K[:2,:] = K[:2,:] * scale

    # read image
    src_img = cv2.imread("/raid/celong/FaceScape/fsmview_images/123/13_lip_funneler/%d.jpg" % test_num)
    src_img = cv2.resize(src_img, (w, h))

    # undistort image
    undist_img = cv2.undistort(src_img, K, dist)

    # read and render mesh
    mesh_dirname = "/raid/celong/FaceScape/fsmview_shapes/123/13_lip_funneler.ply"
    _, rend_img = renderer.render_cvcam(K, Rt, mesh_dirname, std_size=(h, w))

    # project and show
    mix_img = cv2.addWeighted(rend_img, 0.5, undist_img, 0.5, 0)
    concat_img = np.concatenate((undist_img, mix_img, rend_img), axis = 1)
    
    return concat_img


def main():
    cv2.imwrite("./mv_result_49.jpg", projection_test(49, 0.2))
    cv2.imwrite("./mv_result_50.jpg", projection_test(50, 0.2))
    print("results saved to './mv_result_id.jpg'")

if __name__ == "__main__":
    main()
