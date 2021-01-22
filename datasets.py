from utils import resize, resize_dirs, resizev3
from dataset_tool import create_from_images, create_from_image_folders,create_image_and_textv2
from sentence_transformers import SentenceTransformer

class Dataset:
    def __init__(self, path, dim = (512, 512)):
        self.path = path 
        self.dim = dim
        self.encoder = SentenceTransformer('paraphrase-distilroberta-base-v1')
 
    
    def prepare(self, tfrecord_dir, hd5_dir, shuffle = False,
                with_sub_dirs = False, ignore_labels = 1, with_text = True):
        if with_sub_dirs:
            out_path = resize_dirs(self.path, 'dataset/resized', dim = self.dim)
            create_from_image_folders(tfrecord_dir, out_path, shuffle, ignore_labels)
        elif with_text:
            image_file = resizev3(self.path, dim = self.dim)
            create_image_and_textv2(tfrecord_dir, image_file, hd5_dir, shuffle, ignore_labels, self.encoder)
        else:
            print('resizing images ...')
            out_path = resize(self.path, dim = self.dim)
            print('creating records ...')
            create_from_images(tfrecord_dir, out_path, shuffle = True)

