
from training.run_training import run_training
from models import copy_weights, create_model
from utils import download_url
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
from io import BytesIO
import IPython.display
from math import ceil
from PIL import Image, ImageDraw
import os
import pickle
from utils import log_progress, imshow, create_image_grid, show_animation
import imageio
import glob
from sentence_transformers import SentenceTransformer
import h5py
import gensim
from utils import onehot, onehottext

class SGAN:

    def __init__(self, pkl_path = None, init_pkl = 'stylegan2-ffhq-config-f.pkl', from_scratch= False, dim = (512, 512),
                from_dir = None, cond = True, label_size = 768, use_hp5 = False, model_type = 'bert', use_chars = False):
        self.pkl_path = pkl_path
        self.dim = dim
        self.model_type = model_type
        self.use_chars = use_chars

        if model_type == 'bert':
            print('Download Embedding models')
            self.encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    
        if self.pkl_path is None and from_dir is None:
            empty_pkl = create_model(height=dim[0], width=dim[1], cond = cond, label_size = label_size)
        
            if from_scratch:
                self.pkl_path = empty_pkl
            else:
                if init_pkl == 'stylegan2-ffhq-config-f.pkl':
                    ffhq_url = f'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/{init_pkl}'

                    if not os.path.exists(init_pkl):
                        download_url(ffhq_url, init_pkl)
                self.pkl_path = 'surgery.pkl'
                copy_weights(init_pkl, empty_pkl, self.pkl_path)

        if from_dir:
            curr_best = 0
            for pkl_file in glob.glob(f'{from_dir}/*.pkl'):
                ckpt_number = int(pkl_file.split('-')[-1][:-4])
                if curr_best < ckpt_number:
                    curr_best = ckpt_number
                    self.pkl_path = pkl_file

        dnnlib.tflib.init_tf()
        print('Loading networks from "%s"...' % self.pkl_path)
        
        with dnnlib.util.open_url(self.pkl_path) as fp:
            self._G, self._D, self.Gs = pickle.load(fp)
        self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        self.use_hp5 = use_hp5
        self.hd5_dir = 'sample_caption_vectors.hdf5'
        if model_type == 'doc2vec':
            self.encoder = gensim.models.doc2vec.Doc2Vec.load('model.doc2vec')

    def train(self, data_path = None, out_dir = None, mirror = True):
        assert data_path
        assert out_dir
        run_training(data_path, out_dir, resume = self.pkl_path, mirror = mirror)
    
    # Generates a list of images, based on a list of latent vectors (Z), and a list (or a single constant) of truncation_psi's.
    def generate_images_in_w_space(self, dlatents, truncation_psi):
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        Gs_kwargs.truncation_psi = truncation_psi
        # dlatent_avg = self.Gs.get_var('dlatent_avg') # [component]

        imgs = []
        for _, dlatent in log_progress(enumerate(dlatents), name = "Generating images"):
            #row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
            # dl = (dlatent-dlatent_avg)*truncation_psi   + dlatent_avg
            row_images = self.Gs.components.synthesis.run(dlatent,  **Gs_kwargs)
            imgs.append(PIL.Image.fromarray(row_images[0], 'RGB'))
        return imgs       

    def generate_images(self, zs, truncation_psi, class_idx = None):
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if not isinstance(truncation_psi, list):
            truncation_psi = [truncation_psi] * len(zs)
            
        imgs = []
        label = np.zeros([1] + self.Gs.input_shapes[1][1:])
        if class_idx is not None:
            label[:, class_idx] = 1
        else:
            label = None
        for z_idx, z in log_progress(enumerate(zs), size = len(zs), name = "Generating images"):
            Gs_kwargs.truncation_psi = truncation_psi[z_idx]
            noise_rnd = np.random.RandomState(1) # fix noise
            tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in self.noise_vars}) # [height, width]
            images = self.Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
            imgs.append(PIL.Image.fromarray(images[0], 'RGB'))
        return imgs
    
    def generate_images_from_text(self, zs, truncation_psi, text = None):
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if not isinstance(truncation_psi, list):
            truncation_psi = [truncation_psi] * len(zs)
            
        imgs = []
        if self.use_chars:
            text = text.replace(''," ")
        if self.use_hp5:
            with h5py.File(self.hd5_dir, "r") as f:
                # List all groups
                a_group_key = list(f.keys())[0]

                # Get the data
                data = list(f[a_group_key])
                print(len(data))
                text = data[self.idx].reshape((1, 4800))
        elif self.model_type == 'bert':
            text = self.encoder.encode([text])
        elif self.model_type == 'doc2vec':
            text = np.array([self.encoder.infer_vector(text.split())])
        elif self.model_type == 'onehot':
            text = onehot([text])

        elif self.model_type == 'onehottext':
            text = onehottext([text], dim = self.Gs.input_shapes[1][1])
            
        
        for z_idx, z in log_progress(enumerate(zs), size = len(zs), name = "Generating images"):
            Gs_kwargs.truncation_psi = truncation_psi[z_idx]
            noise_rnd = np.random.RandomState(1) # fix noise
            tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in self.noise_vars}) # [height, width]
            images = self.Gs.run(z, text, **Gs_kwargs) # [minibatch, height, width, channel]
            imgs.append(PIL.Image.fromarray(images[0], 'RGB'))
        return imgs
    
    def generate_from_zs(self, zs, truncation_psi = 0.5):
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if not isinstance(truncation_psi, list):
            truncation_psi = [truncation_psi] * len(zs)
            
        for z_idx, z in log_progress(enumerate(zs), size = len(zs), name = "Generating images"):
            Gs_kwargs.truncation_psi = truncation_psi[z_idx]
            noise_rnd = np.random.RandomState(1) # fix noise
            tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in self.noise_vars}) # [height, width]
            images = self.Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            img = PIL.Image.fromarray(images[0], 'RGB')
            imshow(img)

    def generate_random_zs(self, size):
        seeds = np.random.randint(2**32, size=size)
        zs = []
        for _, seed in enumerate(seeds):
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *self.Gs.input_shape[1:]) # [minibatch, component]
            zs.append(z)
        return zs


    def generate_zs_from_seeds(self, seeds):
        zs = []
        for _, seed in enumerate(seeds):
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *self.Gs.input_shape[1:]) # [minibatch, component]
            zs.append(z)
        return zs

    # Generates a list of images, based on a list of seed for latent vectors (Z), and a list (or a single constant) of truncation_psi's.
    def generate_images_from_seeds(self, seeds, truncation_psi, text = None):
        return imshow(self.generate_images_from_text(self.generate_zs_from_seeds(seeds), truncation_psi, text = text)[0])

    def print_G(self):
        self._G.print_layers()

    def print_D(self):
        self._D.print_layers()

    def generate_randomly(self, truncation_psi = 0.5, text = None, idx = 0):
        self.idx = idx
        return self.generate_images_from_seeds(np.random.randint(4294967295, size=1), 
        truncation_psi=truncation_psi, text = text)

    def generate_grid(self, truncation_psi = 0.7): 
      seeds = np.random.randint((2**32 - 1), size=9)
      return create_image_grid(self.generate_images(self.generate_zs_from_seeds(seeds), truncation_psi), 0.7 , 3)
    
    def generate_animation(self, size = 9, steps = 10, trunc_psi = 0.5):
      seeds = list(np.random.randint((2**32) - 1, size=size))
      seeds = seeds + [seeds[0]]
      zs = self.generate_zs_from_seeds(seeds)

      imgs = self.generate_images(self.interpolate(zs, steps = steps), trunc_psi)
      movie_name = 'animation.mp4'
      with imageio.get_writer(movie_name, mode='I') as writer:
        for image in log_progress(list(imgs), name = "Creating animation"):
            writer.append_data(np.array(image))
      return show_animation(movie_name)

    def convertZtoW(self, latent, truncation_psi=0.7, truncation_cutoff=9):
        dlatent = self.Gs.components.mapping.run(latent, None) # [seed, layer, component]
        dlatent_avg = self.Gs.get_var('dlatent_avg') # [component]
        for i in range(truncation_cutoff):
            dlatent[0][i] = (dlatent[0][i]-dlatent_avg)*truncation_psi + dlatent_avg
            
        return dlatent

    def interpolate(self, zs, steps = 10):
        out = []
        for i in range(len(zs)-1):
            for index in range(steps):
                fraction = index/float(steps) 
                out.append(zs[i+1]*fraction + zs[i]*(1-fraction))
        return out
