from utils import resize, resize_dirs, resizev3
from dataset_tool import create_from_images, create_from_image_folders,create_image_and_textv2
from sentence_transformers import SentenceTransformer
import gensim 

class Dataset:
    def __init__(self, path, dim = (512, 512), model_type = 'bert', use_chars = False):
        self.path = path 
        self.dim = dim
        self.model_type = model_type
        self.use_chars = use_chars

        self.encoder = None 
        if model_type == 'bert':
            self.encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        elif model_type == 'doc2vec':
            self.encoder = gensim.models.doc2vec.Doc2Vec.load('model.doc2vec')

 
    
    def prepare(self, tfrecord_dir, text_dir, shuffle = False,
                with_sub_dirs = False, ignore_labels = 1, with_text = True,
                embed_dim = 430):
        if with_sub_dirs:
            out_path = resize_dirs(self.path, 'dataset/resized', dim = self.dim)
            create_from_image_folders(tfrecord_dir, out_path, shuffle, ignore_labels)
        elif with_text:
            image_dir = resize(self.path, dim = self.dim)
            create_image_and_textv2(tfrecord_dir, image_dir, text_dir, shuffle, ignore_labels, self.encoder, 
            model_type = self.model_type, use_chars= self.use_chars, embed_dim = embed_dim)
        else:
            print('resizing images ...')
            out_path = resize(self.path, dim = self.dim)
            print('creating records ...')
            create_from_images(tfrecord_dir, out_path, shuffle = True)

