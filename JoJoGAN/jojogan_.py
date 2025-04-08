import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from util import *
from PIL import Image
import math
import random
import os

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
import wandb
from model import *
from e4e_projection import projection as e4e_projection

from google.colab import files
from copy import deepcopy
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

from e4e.criteria.id_loss import IDLoss

from torchvision.utils import save_image


os.makedirs('inversion_codes', exist_ok=True)
os.makedirs('style_images', exist_ok=True)
os.makedirs('style_images_aligned', exist_ok=True)
os.makedirs('models', exist_ok=True)


device = 'cuda'

def return_generator():
  device = 'cuda' # 만약 cpu로 되어있으면 cuda로 바꿔주세요

  latent_dim = 512

  # Load original generator
  original_generator = Generator(1024, latent_dim, 8, 2).to(device)
  ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
  original_generator.load_state_dict(ckpt["g_ema"], strict=False)
  mean_latent = original_generator.mean_latent(10000)

  # to be finetuned generator
  generator = deepcopy(original_generator)

  return generator

###############################################################################

def return_transform():
  transform = transforms.Compose(
      [
          transforms.Resize((1024, 1024)),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ]
  )
  return transform

###############################################################################

def process_face(img_path:str):

  # Extract filename and name
  filename = img_path.split('/')[-1]
  name = strip_path_extension(img_path) + '.pt'

  # Align and crop face
  aligned_face = align_face(img_path)

  # Perform projection (choose one based on your requirement)
  # my_w = restyle_projection(aligned_face, name, device, n_iters=1).unsqueeze(0)
  my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)

  # Display the aligned face
  # display_image(aligned_face, title='Aligned face')

  return aligned_face, my_w

#################################################################################

def get_target_im(img_path : str):
    # Upload your own style images into the style_images folder and type it into the field in the following format without the directory name. Upload multiple style images to do multi-shot image translation
  style_path = img_path

  targets = []
  latents = []

  assert os.path.exists(style_path), f"{style_path} does not exist!"

  name = style_path.split("/")[-1]
  name = strip_path_extension(name)

  # crop and align the face
  style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')
  if not os.path.exists(style_aligned_path):
      style_aligned = align_face(style_path)
      style_aligned.save(style_aligned_path)
  else:
      style_aligned = Image.open(style_aligned_path).convert('RGB')

  # GAN invert
  style_code_path = os.path.join('inversion_codes', f'{name}.pt')
  if not os.path.exists(style_code_path):
      latent = e4e_projection(style_aligned, style_code_path, device)
  else:
      latent = torch.load(style_code_path)['latent']

  transform = return_transform()

  targets.append(transform(style_aligned).to(device))
  latents.append(latent.to(device))

  targets = torch.stack(targets, 0)
  latents = torch.stack(latents, 0)

  target_im = utils.make_grid(targets, normalize=True, value_range=(-1, 1))

  return targets, target_im, latents

##########################################################################

def return_discriminator():
  device = "cuda"
  # load discriminator for perceptual loss
  discriminator = Discriminator(1024, 2).eval().to(device)
  ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
  discriminator.load_state_dict(ckpt["d"], strict=False)

  return discriminator

########################################################################
def return_ID_loss():
  from e4e.criteria.id_loss import IDLoss 

  device = "cuda"

  identity_loss = IDLoss()
  identity_loss = identity_loss.to(device)

  return identity_loss

###########################################################################

def from_toon_2_real(project_direc:str, episode_num:int, hyper_param:tuple, real_name:str):
  num_iter, alpha, loss_multiplier = hyper_param

  device = "cuda"
  latent_dim = 512

  curr_direc = os.path.dirname(os.path.abspath("Project"))
  curr_direc = os.path.join("/",*curr_direc.split("/")[1:-1])

  result_direc = os.path.join(curr_direc,"result")
  result_episode_direc = os.path.join(result_direc, str(episode_num))
  os.makedirs(result_episode_direc, exist_ok=True)

  result_episode_cropped_face_direc = os.path.join(result_episode_direc,"cropped_faces")
  os.makedirs(result_episode_cropped_face_direc, exist_ok=True)

  result_episode_style_transferred_images = os.path.join(result_episode_direc,"style_transferred_images")
  os.makedirs(result_episode_style_transferred_images, exist_ok=True)
  
  cropped_faces_list = [i for i in os.listdir(result_episode_cropped_face_direc) if i != ".DS_Store" and i != ".ipynb_checkpoints"]


  # 진짜 사람 얼굴 가져와서 process 하기
  real_direc = os.path.join(curr_direc, "real_faces")
  real_img_filename = [i for i in os.listdir(real_direc) if real_name in i][0]
  real_img_path = os.path.join(real_direc, real_img_filename)

  aligned_face, my_w = process_face(img_path = real_img_path)

  aligned_face_tensor = np.array(aligned_face)
  aligned_face_tensor = aligned_face_tensor.astype(np.float32)
  aligned_face_tensor /= 255.0 # 이렇게 안 하면 색이 반전된 이미지가 올라가서...
  aligned_face_tensor = torch.tensor(aligned_face_tensor).permute(2, 0, 1).unsqueeze(0).float()
  aligned_face_tensor = aligned_face_tensor.to(device)

  identity_loss = return_ID_loss()
  original_generator = return_generator()
  generator = deepcopy(original_generator)


  for cropped_face_file_name in cropped_faces_list:

    cropped_face_path = os.path.join(result_episode_cropped_face_direc, cropped_face_file_name)
    targets, target_im, latents = get_target_im(cropped_face_path)
    target_im = target_im.unsqueeze(0)

    preserve_color = False

    discriminator = return_discriminator()
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    discriminator.load_state_dict(ckpt["d"], strict=False)
    
    del generator
    generator = deepcopy(original_generator)
    g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0.0, 0.99))

    if preserve_color:
      id_swap = [9,11,15,16,17]
    else:
        id_swap = list(range(7, generator.n_latent))

    checo = 0
    for idx in tqdm(range(num_iter)):
        mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        in_latent = latents.clone()
        in_latent[:, id_swap] = alpha*latents[:, id_swap] + (1-alpha)*mean_w[:, id_swap]

        img = generator(in_latent, input_is_latent=True)
        checo = img

        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(img)

        # 기존의 perception loss
        perception_loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)
        # 위에서 초기화한 identity loss
        id_loss, _, _ = identity_loss.forward(img, target_im, aligned_face_tensor)

        
        total_loss = id_loss * loss_multiplier + perception_loss

        g_optim.zero_grad()
        total_loss.backward()
        g_optim.step()

    seed = 3000

    torch.manual_seed(seed)
    with torch.no_grad():
        generator.eval()

        original_my_sample = original_generator(my_w, input_is_latent=True)
        my_sample = generator(my_w, input_is_latent=True)


    face = aligned_face_tensor.permute(0, 1, 2, 3)

    my_output = my_sample
 
    crop_name_wo_extension = cropped_face_file_name.split(".")[0]
    new_name = f"{crop_name_wo_extension}.st.png"

    output_path = os.path.join(result_episode_style_transferred_images,new_name)

    save_image(my_output, output_path)

