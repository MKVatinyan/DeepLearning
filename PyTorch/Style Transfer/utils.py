from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
import torch

# Small custom method to show images neatly
def imshow_noaxis(data, title = '', axis = 'off'):
    
    if type(data) == torch.Tensor:
        data = data.squeeze(0).permute(1, 2, 0).detach().numpy()
        imshow_noaxis(data, title)

    plt.imshow(data)
    plt.title(title)
    plt.axis(axis)   
    
    
def create_and_save_white_noise_image(w, h, path):
    
    noise = np.zeros((h, w, 3), 'uint8')
    random_matrix = np.random.randint(0, 255, size = (h, w)).astype(np.uint8)
    noise[...,0] = noise[...,1] = noise[...,2] = random_matrix
    Image.fromarray(noise).save(path)
    
def pil_image_to_torch_tensor(image):
    preprocess = transforms.Compose([
                        transforms.Resize(300),
                        transforms.ToTensor()])
                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return preprocess(image).unsqueeze(0)