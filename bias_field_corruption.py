import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image

def load_image_as_tensor(image_path): 
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    return transform(image)


def generate_bias_field(bB, img_size_target):
    '''
    generating a bias field that's the same size as the img
    
    input:
        bB: hyperparameter to control the bias field standard deviation range
        img_size_target: a tuple of (img_width_pixels, img_height_pixels)
        
    output: biased field
    '''
    
    
    # std of the biased field --> sigmaB ~ U(0, bB)
    sigmaB = torch.rand(1) * bB
        
    # 4x4 image sampled from a Gaussian with mean 0 and std σB --> B' = N_4x4(0, sigmaB^2)
    bias_field_small = torch.randn((1, 1, 4, 4)) * sigmaB
        
    # upsapling bias_field_small to the image size using interpolation
    bias_field = F.interpolate(bias_field_small, size=img_size_target, mode='bicubic', align_corners=False)
    
    return bias_field.squeeze() # make shape [img_size_width, img_size_height]
    


def apply_bias_field(G, bias_field):
    '''
    applying biased field onto img G to obtain a biased img
    '''
    G_float = G.unsqueeze(0).float() if G.dim() == 2 else G.float()
    GB = G_float * torch.exp(bias_field)
    
    return GB.squeeze()  



def rescale_image(image):
    '''
    rescale img to have all values to be in between 0 and 1
    '''
    min_val = image.min()
    max_val = image.max()
    rescaled_image = (image - min_val) / (max_val - min_val)
    return rescaled_image



def apply_gamma_transform(image, sigma_gamma):
    '''
    applying a random Gamma transform to the img
    '''
    # sample gamma from a Gaussian with mean 0 and std sigma_gamma
    gamma = torch.normal(mean=torch.zeros(1), std=sigma_gamma * torch.ones(1))
    
    # apply the Gamma transform
    gamma_image = image.pow(gamma.exp())

    return gamma_image



def process_images_in_folder(folder_path, output_folder, bB, sigma_gamma):
    '''
    processing imgs in the training folder
        - bB is the hyperparameter that controls the range of possible bias 
          field std values sampled from an uniformed distribution
        - sigma_gamma is another hyperparameter that controls the intensity of 
          the output HR image
    
    '''
    
    folder_path = Path(folder_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for image_path in folder_path.glob('*generation.tif'):
        # loading img
        G = load_image_as_tensor(image_path)
        
        # if img not grayscale, convert it to grayscale
        if G.shape[0] > 1:
            G = G.mean(dim=0, keepdim=True)

        bias_field = generate_bias_field(bB, (G.shape[1], G.shape[2]))
        GB = apply_bias_field(G, bias_field)
        G_rescaled = rescale_image(GB)
        I_HR = apply_gamma_transform(G_rescaled, sigma_gamma)
        
        # saving
        save_path = output_folder / f'{image_path.stem}_biased.tif'
        save_image(I_HR, save_path)
        print(f"Processed and saved: {save_path}")
        
        
        
        
folder_path = '/Users/ariellin/Documents/EC 500/Data/PROJECT_DATA/training'
output_folder = '/Users/ariellin/Documents/EC 500/Data/fake_img2' 
bB = 0.5  
sigma_gamma = 0.5

process_images_in_folder(folder_path, output_folder, bB, sigma_gamma)

