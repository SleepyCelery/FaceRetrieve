from facenet import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

if __name__ == '__main__':
    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    face1 = Image.open('./test_image/face_00001.jpg')
    face2 = Image.open('./test_image/face_00002.jpg')

    img_cropped1 = mtcnn(face1, save_path='./cropped_image/face_00001.jpg')
    img_cropped2 = mtcnn(face2, save_path='./cropped_image/face_00002.jpg')

    face1_embedding = resnet(img_cropped1.unsqueeze(0))
    face2_embedding = resnet(img_cropped2.unsqueeze(0))

    print(face1_embedding)
    print(face2_embedding)

    print(np.linalg.norm(face1_embedding.detach().numpy() - face2_embedding.detach().numpy()))
