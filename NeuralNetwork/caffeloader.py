# This file is data load for pytorch, but following the input format of caffe library.
# The number of classes is not required for label conversion.
# Two arguments are needed to load image files: one is the root of directory,
#                                               the other is a txt file containing filepath and label.
# The created ImageFile object can be passed to a pytorch DataLoader for multi-threading process.
#
# Author : Sen Jia 
#

import torch.utils.data as data

from PIL import Image
import os
import os.path


#IMG_EXTENSIONS = [
#    '.jpg', '.JPG', '.jpeg', '.JPEG',
#    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
#]


#def is_image_file(filename):
#    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def make_dataset(root,txt_file):
    images = []
    with open(txt_file,"r") as f:
        for line in f:
            strs = line.rstrip("\n").split(" ")
            images.append((os.path.join(root,strs[0]),int(strs[1])))
    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


#def default_loader(path):
#    from torchvision import get_image_backend
#    if get_image_backend() == 'accimage':
#        return accimage_loader(path)
#    else:
#        return pil_loader(path)

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageList(data.Dataset):
    def __init__(self, root, txt_file, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(root, txt_file)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

