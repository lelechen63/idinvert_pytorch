# python 3.6
"""Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import pickle
from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name', type=str, help='Name of the GAN model.')
  # parser.add_argument('image_list', type=str,
  #                     help='List of images to invert.')
  # parser.add_argument('-o', '--output_dir', type=str, default='',
  #                     help='Directory to save the results. If not specified, '
  #                          '`./results/inversion/${IMAGE_LIST}` '
  #                          'will be used by default.')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--num_results', type=int, default=1,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
  parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
  parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                      help='The encoder loss scale for optimization.'
                           '(default: 2.0)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()




def load_data():
    """ load the video data"""
    img_path = '/home/cxu-serve/p1/lchen63/nerf/data/mead/001/aligned'
    img_names = os.listdir(img_path)
    img_names.sort()
    gt_imgs = []
    f = open('/home/cxu-serve/p1/lchen63/nerf/data/mead/001/img_list.txt','w')
    for i in range(len(img_names)):
        # if i == 4:
        #     break
        img_p = os.path.join( img_path, img_names[i])
        f.write(img_p +'\n')       
    f.close
    return gt_imgs


def main_load_data():
    """ load the video data"""
    base_p = '/raid/celong/FaceScape/ffhq_aligned_img'
    _file = open( '/raid/celong/lele/github/idinvert_pytorch/predef/frontface_list.pkl', "rb")
    valid_all = pickle.load(_file)
    ids =  os.listdir(base_p)
    ids.sort()
    img_names = []
    print (valid_all)
    for id_p in ids:
        current_p = os.path.join( base_p , id_p)
        
        for motion_p in os.listdir(current_p):
            current_p1 = os.path.join( current_p , motion_p)
            # try:
              # print (id_p +'__' + motion_p)
            valid_idxs = valid_all[id_p +'__' + motion_p]
            for valid_f in valid_idxs:
              img_path = os.path.join( current_p1, valid_f + '.jpg')
              img_names.append(img_path)
              print (img_path)
            # except:
              
            #   continue
    img_names.sort()
    f = open( os.path.join('/raid/celong/FaceScape', 'lists', 'inversion.txt'),'w')
    for i in range(len(img_names)):
        img_p = os.path.join( img_path, img_names[i])
        f.write(img_p +'\n')
        # if i == 5:
        #   break   
    f.close
    return  os.path.join('/raid/celong/FaceScape', 'lists', 'inversion.txt')


def main(image_list = None):
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.exists(image_list)

  image_list_name = os.path.splitext(os.path.basename(image_list))[0]

  output_dir = f'/raid/celong/lele/tmp/inversion/{image_list_name}'

  # output_dir = args.output_dir or f'results/inversion/{image_list_name}'
  logger = setup_logger(output_dir, 'inversion.log', 'inversion_logger')

  logger.info(f'Loading model.')
  inverter = StyleGANInverter(
      args.model_name,
      learning_rate=args.learning_rate,
      iteration=args.num_iterations,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=args.loss_weight_feat,
      regularization_loss_weight=args.loss_weight_enc,
      logger=logger)
  image_size = inverter.G.resolution

  # Load image list.
  logger.info(f'Loading image list.')
  img_list = []
  with open(image_list, 'r') as f:
    for line in f:
      img_list.append(line.strip())
  # Initialize visualizer.
  save_interval = args.num_iterations // args.num_results
  headers = ['Name', 'Original Image', 'Encoder Output']
  for step in range(1, args.num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:06d}')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=len(img_list), num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  # Invert images.
  logger.info(f'Start inversion.')
  latent_codes = []
  for img_idx in tqdm(range(len(img_list)), leave=False):
    image_path = img_list[img_idx]
    image_name = image_path.split('/')[-3] +'__' + image_path.split('/')[-2] + '__' +image_path.split('/')[-1][:-4]
    mask_path = image_path[:-4] +'_mask.png'
    # if  os.path.exists (image_path[:-3] +  'npy'):
    #   print (image_path[:-3] +  'npy')
    #   print ('!!!')
      # continue
    try:
      image = load_image(image_path)
      mask = cv2.imread(mask_path)
      print (mask.shape, image.shape)
      image = image * mask 
      image = resize_image(image, (image_size, image_size))
      code, viz_results = inverter.easy_invert(image, num_viz=args.num_results)
      latent_codes.append(code)
      np.save(image_path[:-3] +  'npy',code)
      save_image(f'{output_dir}/{image_name}__ori.png', image)
      save_image(f'{output_dir}/{image_name}__enc.png', viz_results[1])
      save_image(f'{output_dir}/{image_name}__inv.png', viz_results[-1])
      visualizer.set_cell(img_idx, 0, text=image_name)
      visualizer.set_cell(img_idx, 1, image=image)
      for viz_idx, viz_img in enumerate(viz_results[1:]):
        visualizer.set_cell(img_idx, viz_idx + 2, image=viz_img)
    except:
      continue
  # Save results.
  # os.system(f'cp {args.image_list} {output_dir}/image_list.txt')
  visualizer.save(f'{output_dir}/inversion.html')


if __name__ == '__main__':
  img_list = main_load_data()
  main(img_list)
