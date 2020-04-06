#coding:utf-8
import numpy as np
import tensorflow as tf
import random
from PIL import Image
from captcha.image import ImageCaptcha  # pip install captcha
import matplotlib.pyplot as plt

dataset = ['0','1','2','3','4','5','6','7','8','9',\
	'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',\
	'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
def create_captcha(size = 4):
    captcha_text = []
    for i in range(size):
    	c = random.choice(dataset)
    	captcha_text.append(c)
    captcha_text = ''.join(captcha_text)

    image = ImageCaptcha()
    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image.convert('L'),'f')
    return captcha_image,captcha_text

def parse_image(images):
    plt.matshow(images,cmap = plt.get_cmap('gray'))
    plt.savefig("image.png")

def vec_pos(c):
    ascii = ord(c)
    if ascii >= 0x61:
        return ascii - 0x61 + 36
    elif ascii >= 0x41:
        return ascii - 0x41 + 10
    else:
        return ascii - 0x30

def str2vec(str,size = 4):
    vector = np.zeros(len(dataset)*size)
    for i, c in enumerate(str):
        idx = i * len(dataset) + vec_pos(c)
        vector[idx] = 1
    return vector

def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_idx = c % len(dataset)
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

def create_batch(batch_size):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, len(dataset)*4])

    def create_text_and_image():
        while True:
            image,text = create_captcha()
            if image.shape == (60,160):
                return text,image
    
    for i in range(batch_size):
        text,image = create_text_and_image()
        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0
        batch_y[i,:] = str2vec(text)
    return batch_x,batch_y
    
if __name__ == '__main__':
    create_batch(1)