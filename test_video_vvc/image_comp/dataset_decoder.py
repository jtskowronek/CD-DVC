# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py

import os
import os.path

import torch
import random
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def crop_cv2(img, patch, i):
    height, width, c = img.shape
    
    start_x = random.randint(0, height - patch)
    start_y = random.randint(0, width - patch)
    #print(start_x,start_y)
    return img[start_x : start_x + patch, start_y : start_y + patch]

# 将数据类型转化为tensor
def np_to_torch(img):
    img = np.swapaxes(img, 0, 1) #1, w, h, 3
    img = np.swapaxes(img, 0, 2) #1, 3, h, w
    return torch.from_numpy(img).float()


def default_loader(path):
    return Image.open(path).convert('L')


class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, is_train, root, transform=None, loader=default_loader):
        images = []
        images_pre = []
        images_next = []
        # for filename in os.listdir(root):
        #     if is_image_file(filename):
        #         images.append('{}'.format(filename))

        if is_train :  # 训练用viemo

            for video in os.listdir(root):
                if not is_image_file(video):
                    for frame in os.listdir(root + "/" + video):
                        if not is_image_file(frame):
                            image = os.listdir(root + "/" + video + "/" + frame)
                            if is_image_file(image[1]):
                                images.append('{}'.format(video + "/" + frame + "/" + image[1]))
                                images_pre.append('{}'.format(video + "/" + frame + "/" + image[0]))
                                images_next.append('{}'.format(video + "/" + frame + "/" + image[2]))

        else:  # 测试用foreman
            files = os.listdir(root)
            files.sort(key= lambda x:int(x[-9:-4]))
            for filename in files:
                if is_image_file(filename):
                    #print(filename)
                    images.append('{}'.format(filename))
        #print(images)
        self.is_train = is_train
        self.root = root
        self.imgs = images
        self.imgs_pre = images_pre
        self.img_next = images_next
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index+1]
        if self.is_train == False :
            sorted(self.imgs, key=lambda x:filename.split('_')[2].split('.')[0])

        
        if self.is_train:  # viemo
            filenamePre = self.imgs_pre[index]
            filenameNext = self.img_next[index]
            
        else:  # 测试集为foreman
            i = index+1
            if i == 0:
                filenamePre = filename
                filenameNext = self.imgs[i + 1]
            elif index == len(self.imgs) - 1 :
                filenamePre = self.imgs[i - 1]
                filenameNext = filename
            else:
                filenamePre = self.imgs[i - 1]
                if filenamePre.split('_')[0] != filename.split('_')[0]: # 分辨是不是同一个视频数据集
                    filenamePre = filename

                filenameNext = self.imgs[ + 1]
                if filenameNext.split('_')[0] != filename.split('_')[0]:
                    filenameNext = filename

            

        try:
            #print(os.path.join(self.root, filename))
            img = self.loader(os.path.join(self.root, filename))
            imgPre = self.loader(os.path.join(self.root, filenamePre))
            imgNext = self.loader(os.path.join(self.root, filenameNext))
            imgPre_array = np.asarray(imgPre)
            img_array = np.asarray(img)
            imgNext_array = np.asarray(imgNext)
            imgPre_array = np.expand_dims(imgPre_array, axis=2)
            img_array = np.expand_dims(img_array, axis=2)
            imgNext_array = np.expand_dims(imgNext_array, axis=2)
            imgAll = np.concatenate((imgPre_array, img_array, imgNext_array), axis=2)
        except:
            print('error')
            return torch.zeros((9, 64, 64))
        

        if self.is_train:  # 训练时，将图片剪裁为两个64*64的块
            
            crops = []
            for i in range(2):
                #print(i)
                data = crop_cv2(imgAll, 64, i)  # 随机截取大小为[64,64,3]的矩阵
                data = data / 255.0  # 归一化
                crops.append(np_to_torch(data))
            data = crops
        else:  # 测试时，保持图片的原大小
            img = imgAll / 255.0
            data = np_to_torch(img)
        #print("t")
        return data, filename, filenamePre, filenameNext

    def __len__(self):
        return len(self.imgs)-1
