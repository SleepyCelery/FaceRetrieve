from PIL import Image
from facenet import MTCNN, InceptionResnetV1
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()


def extract(image_path):
    image = Image.open(image_path)
    image_cropped = mtcnn(image)
    if image_cropped is not None:
        image_embedding = resnet(image_cropped.unsqueeze(0).cuda())
        image_embedding_np = image_embedding.cpu().detach().numpy()
        return image_embedding_np
    else:
        return None
