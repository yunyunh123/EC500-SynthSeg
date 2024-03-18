import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from pathlib import Path



def load_image_as_tensor(image_path): 
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    return transform(image)

def simulate_resolution(image, range_res):
    # making the image shape [B, C, H, W]. previously [1, H, W]
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # sample from uniform distribution [1mm, 9mm] for LR spacing
    r_spac = torch.FloatTensor(1).uniform_(*range_res).item()

    scale_factor = 1 / r_spac
    downsampled_image = F.interpolate(image, scale_factor=scale_factor, mode='area')
    upsampled_image = F.interpolate(downsampled_image, size=(image.size(2), image.size(3)), mode='bicubic', align_corners=False)

    return upsampled_image


# def simulate_resolution_wG(image, range_res, range_alpha):
#     # making the image shape [B, C, H, W]. previously [1, H, W]
#     if image.dim() == 3:
#         image = image.unsqueeze(0)

#     # sample from uniform distribution [1mm, 9mm] for LR spacing
#     r_spac = torch.FloatTensor(1).uniform_(*range_res).item()
    
#     # sample alpha from U(a_a, b_a)
#     alpha = torch.FloatTensor(1).uniform_(*range_alpha).item()
    
#     sigma = 
#     blurred_image = 
    
#     # Downsample and then upsample to simulate change in resolution
#     downsampled_image = F.interpolate(blurred_image, scale_factor=1/r_spac, mode='area')
#     upsampled_image = F.interpolate(downsampled_image, scale_factor=(image.size(2), image.size(3)), mode='bicubic', align_corners=False)

#     return upsampled_image.squeeze() 


def process_images_in_folder(folder_path, output_folder, range_res):
    
    folder_path = Path(folder_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for image_path in folder_path.glob('*generation_biased.tif'):
        img = load_image_as_tensor(image_path)
        
        if img.shape[0] > 1:
            img = img.mean(dim=0, keepdim=True)

        stimulated_img = simulate_resolution(img, range_res)
        
        # saving
        save_path = output_folder / f'{image_path.stem}_sampled.tif'
        save_image(stimulated_img, save_path)
        print(f"Processed and saved: {save_path}")


range_res = (1, 3)  # low resolution parameter range [1mm, 9mm]  maybe range too big??
# a_a = 0.9   
# b_a = 1.1  


folder_path = '/Users/ariellin/Documents/EC 500/Data/fake_img2'
output_folder = '/Users/ariellin/Documents/EC 500/Data/fake_img3' 

process_images_in_folder(folder_path, output_folder, range_res)



