from PIL import Image
from facenet import MTCNN, InceptionResnetV1
import os
import database
import time

total_num = 136719
offset = 0
device = 0

if __name__ == '__main__':
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    # 136719 images totally, the filename of first image is 000000_0.jpg and the last is 152495_0.jpg

    base_dir = './facesets/'
    filename_list = [os.path.join(base_dir, '{:0>6d}.jpg'.format(x)) for x in range(0, total_num)]

    database_record = {}
    starttime = time.time()

    #  The data dump in database should include these attrs:
    #  _id(index begin from 0)
    #  filename
    #  feature_vector(numpy vector/list)
    for index, filename in enumerate(filename_list[offset:], offset):
        database_record['_id'] = index
        database_record['filename'] = filename
        with open(filename, mode='rb') as file:
            image = Image.open(file)
            image_cropped = mtcnn(image)
            if image_cropped is not None:
                image_embedding = resnet(image_cropped.unsqueeze(0).cuda())
                image_embedding_np = image_embedding.cpu().detach().numpy()
                database_record['feature_vector'] = image_embedding_np.tolist()

            else:
                database_record['feature_vector'] = None
                print('Image at index {} extract face failed!'.format(index))

            database.add_face(database_record)
            print('Progress: {}/{}'.format(index + 1, total_num))

    costtime = time.time() - starttime
    print('Cost time: {}s, Average time per image: {}s'.format(costtime, costtime / (total_num - offset)))
