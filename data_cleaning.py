import os

# 136719 images totally, the filename of first image is 000000_0.jpg and the last is 152495_0.jpg
fileindex_max = 136719
base_dir = './facesets/'
count = 0

if __name__ == '__main__':
    for filename in os.listdir(base_dir):
        if count < fileindex_max:
            os.rename(os.path.join(base_dir, filename), os.path.join(base_dir, '{:0>6d}.jpg'.format(count)))
            count += 1
