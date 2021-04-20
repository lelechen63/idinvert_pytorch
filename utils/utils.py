import cv2
import PIL
import PIL.Image
import sys
import os
import glob
import scipy
import scipy.ndimage
import dlib
import numpy as np
import time 


# download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# predictor = dlib.shape_predictor('/u/lchen63/github/genforce/utils/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor('/raid/celong/lele/github/idinvert_pytorch/utils/shape_predictor_68_face_landmarks.dat')
def get_landmark(filepath):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))


    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    # lm is a shape=(68,2) np.array
    return lm

    
def align_face(filepath, output_path, landmark_path = None ):  
    """
    transfer image to image aligned with ffhq dataset.
    :param filepath: str
    :return: PIL Image
    """
    a = time.time()
    # if landmark_path is None:
    lm = get_landmark(filepath)
    # else:
    #     lm = np.load(landmark_path)
    #     lm = np.transpose(lm, (1, 0))
    # lm = np.load(landmark_path).transpose(1,0)[:,::-1]

    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    
    # read image
    img = PIL.Image.open(filepath)

    output_size=1024
    transform_size=4096
    enable_padding=True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        # img.save('./gg.png'  )
        # img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        # img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        # img.save('./gg2.png'  )

        # print (time.time() - d, '+++1')
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        # print (time.time() - d, '++2z')
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
        # print (time.time() - d)


    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
   
    # Save aligned image.
    # img.save('./gg3.png'  )
    print (time.time() - a)
    img.save(output_path  )
    # return img

def trans_video_to_imgs( video_path, save_img_folder, write_img = True ):
    video_cap = cv2.VideoCapture(video_path)
    if video_cap.isOpened() == False:
        print('Error in opening ' + video_path)
        return -1

    frm_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not write_img:
        return frm_num

    ret, frm = video_cap.read()
    frm_id = 0
    while ret is True:
        save_img_path = save_img_folder + '/image_%05d.png' % frm_id

        cv2.imwrite(save_img_path, frm)

        align_face(save_img_path, save_img_path)

        ret, frm = video_cap.read()
        frm_id += 1
        print(' Save frame %d/%d' % ( frm_id, frm_num ), end='\r', flush=True)
    print( '' )

    return frm_num

def main_mead_video2imgs():
    base_p = '/home/cxu-serve/p1/lchen63/nerf/data/mead/video'
    for view_p in os.listdir(base_p):
        current_p = os.path.join( base_p , view_p)
        for motion_p in os.listdir(current_p):
            current_p1 = os.path.join( current_p , motion_p)
            for level_p in os.listdir(current_p1):
                current_p2 = os.path.join( current_p1 , level_p)
                for v_id in os.listdir(current_p2):
                    if v_id[-4:] =='.mp4':
                        v_p =  os.path.join( current_p2 , v_id)
                        if not os.path.exists(v_p[:-4]):
                            os.mkdir( v_p[:-4] )
                        trans_video_to_imgs( v_p, v_p[:-4] , write_img = True )

def main_facescape_align():
    base_p = '/raid/celong/FaceScape/fsmview_images'
    if not os.path.exists( base_p.replace('fsmview_images', 'ffhq_aligned_img') ):
        os.mkdir(base_p.replace('fsmview_images', 'ffhq_aligned_img'))
    save_p = base_p.replace('fsmview_images', 'ffhq_aligned_img')
    for id_p in os.listdir(base_p):
        current_p = os.path.join( base_p , id_p)
        save_p1 = os.path.join( save_p , id_p)
        if not os.path.exists(  os.path.join( save_p1 ) ):
            os.mkdir( save_p1 ) 
        for motion_p in os.listdir(current_p):
            current_p1 = os.path.join( current_p , motion_p)
            save_p2 = os.path.join( save_p1 , motion_p)
            if not os.path.exists(  os.path.join( save_p2 ) ):
                os.mkdir( save_p2 ) 
            img_names = os.listdir(current_p1)
            img_names.sort()
            for i in range(len(img_names)):
                img_p = os.path.join( current_p1, img_names[i])
                output_p = os.path.join( save_p2 , img_names[i])
                # lmark_p = img_p.replace('fsmview_images', 'fsmview_landmarks')[:-3] +'npy'
                # if os.path.exists(output_p):
                #     continue
                try:
                    align_face(img_p, output_p)
                    align_face(img_p, output_p, lmark_p)
                    print (output_p)
                except:
                    print (img_p , lmark_p)
                    continue
            #     aligned_img = cv2.imread(img_p.replace( 'original', 'aligned'))
            #     aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR)
            #     gt_imgs.append(preprocess(aligned_img))
            # gt_imgs = np.asarray(gt_imgs)
            # gt_imgs = torch.FloatTensor(gt_imgs)
            # return gt_imgs




def load_data():
    """ load the video data"""
    img_path = '/home/cxu-serve/p1/lchen63/nerf/data/mead/001/original'
    img_names = os.listdir(img_path)
    img_names.sort()
    gt_imgs = []
    for i in range(len(img_names)):
        if i == 4:
            break
        img_p = os.path.join( img_path, img_names[i])
        # align_face(img_p)
        aligned_img = cv2.imread(img_p.replace( 'original', 'aligned'))
        aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR)
        # print (aligned_img.shape)
        gt_imgs.append(preprocess(aligned_img))
    gt_imgs = np.asarray(gt_imgs)
    gt_imgs = torch.FloatTensor(gt_imgs)
    return gt_imgs

    
main_facescape_align()
# trans_video_to_imgs( '/raid/celong/mead/tmp/001.mp4', '/raid/celong/mead/tmp/001', write_img = True )
# trans_video_to_imgs( '/home/cxu-serve/p1/lchen63/nerf/data/mead/001.mp4', '/home/cxu-serve/p1/lchen63/nerf/data/mead/001/original', write_img = True )