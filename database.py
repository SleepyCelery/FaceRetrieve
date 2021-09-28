import numpy as np
import cupy as cp
import pymongo
import time
from tqdm import tqdm

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['facesets']

faces = db['faces']


#  The data dump in database should include these attrs:
#  _id(index begin from 0)
#  filename
#  feature_vector(numpy vector/list)


#  use tolist() method to convert numpy.ndarray to list for dumping into mongodb (1 dimension array)
#  use numpy.fromiter() method to read ndarray_like list stored in mongodb (1 dimension array)

def add_face(info: dict):
    if isinstance(info['feature_vector'], numpy.ndarray):
        info['feature_vector'] = info['feature_vector'].tolist()
    x = faces.insert_one(info)
    print(x.inserted_id)


def add_faces(info: list):
    for i in range(0, len(info)):
        if isinstance(info[i]['feature_vector'], numpy.ndarray):
            info[i]['feature_vector'] = info[i]['feature_vector'].tolist()
    x = faces.insert_many(info)
    print(x.inserted_ids)


def search_face_by_index(index: int):
    x = faces.find_one({'_id': index})
    return dict(x)


def build_feature_mat(cuda=False):
    total_length = faces.count()
    result = []
    count = 0
    with tqdm(total=100) as pbar:
        for x in faces.find():
            count += 1
            if x['feature_vector'] is not None:
                result.append(x['feature_vector'][0])
            else:
                result.append([10 for _ in range(512)])
            if count >= total_length / 100:
                pbar.update(1)
                pbar.set_description('Reading features from database ')
                count = 0
    print('convert features to numpy/cupy matrix...')
    if cuda:
        result = cp.array(result).swapaxes(0, 1)
    else:
        result = np.array(result).swapaxes(0, 1)

    return result


if __name__ == '__main__':
    starttime = time.time()
    print(build_feature_mat(cuda=True).shape)
    print(time.time() - starttime)
