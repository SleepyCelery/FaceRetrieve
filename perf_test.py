import time
import numpy as np
import cupy as cp
import database
import feature_extract
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

data = database.build_feature_mat(cuda=True)


def face_retrieve(image_path: str, cuda=False, show=False):
    starttime = time.time()
    if cuda:
        try:
            target = cp.asarray(feature_extract.extract(image_path)).swapaxes(1, 0)
        except:
            print('Extract features failed! Image path {}'.format(image_path))
            return None
        stage1_time = time.time()

        sub = data - target
        stage2_time = time.time()

        result = cp.linalg.norm(sub, axis=0)
        stage3_time = time.time()

        similar_index = cp.argsort(result, axis=0)[:3]  # get top-3 similar faces
        stage4_time = time.time()
    else:
        try:
            target = np.asarray(feature_extract.extract(image_path)).swapaxes(1, 0)
        except:
            print('Extract features failed! Image path {}'.format(image_path))
            return None
        stage1_time = time.time()

        sub = data - target
        stage2_time = time.time()

        result = np.linalg.norm(sub, axis=0)
        stage3_time = time.time()

        similar_index = np.argsort(result, axis=0)[:3]  # get top-3 similar faces
        stage4_time = time.time()
    print(
        'Cost time:{}s, read image cost {}s, sub matrix cost {}s, calc distance cost {}s, data sort cost {}s.'.format(
            stage4_time - starttime, stage1_time - starttime, stage2_time - stage1_time, stage3_time - stage2_time,
            stage4_time - stage3_time))
    if show:
        plt.subplot(2, 2, 1), plt.title('Original')
        image1 = Image.open(image_path)
        plt.imshow(image1), plt.axis('off')

        plt.subplot(2, 2, 2), plt.title('1(index:{})'.format(int(similar_index[0])))
        image2 = Image.open(database.search_face_by_index(int(similar_index[0]))['filename'])
        plt.imshow(image2), plt.axis('off')

        plt.subplot(2, 2, 3), plt.title('2(index:{})'.format(int(similar_index[1])))
        image3 = Image.open(database.search_face_by_index(int(similar_index[1]))['filename'])
        plt.imshow(image3), plt.axis('off')

        plt.subplot(2, 2, 4), plt.title('3(index:{})'.format(int(similar_index[2])))
        image4 = Image.open(database.search_face_by_index(int(similar_index[2]))['filename'])
        plt.imshow(image4), plt.axis('off')

        plt.show()


if __name__ == '__main__':
    basedir = './facesets'
    test_image = [os.path.join(basedir, '{:0>6}.jpg'.format(x)) for x in random.sample(range(0, 136719), 20)]
    for image in test_image:
        face_retrieve(image, cuda=True, show=True)
    # face_retrieve(os.path.join(basedir, '000000.jpg'), cuda=True, show=True)
