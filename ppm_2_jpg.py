from PIL import Image
import os


def convert(dirfrom, dirto):
    names = os.listdir(dirfrom)
    for file in names:
        im = Image.open(dirfrom + file)
        im.save(dirto + file.split('.')[0] + '.jpg')


dirfrom = '/home/gabi/Documents/datasets/humans/png_0/'
dirto = '/home/gabi/Documents/datasets/humans/0/'

convert(dirfrom, dirto)