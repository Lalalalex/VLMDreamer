import torch
from diffusers import FluxImg2ImgPipeline
from ops.utils.general import convert_PIL
import numpy as np

def convert_PIL(image):
    """
        image can be a PIL Image, a path to an image, a numpy array ([H, W, c]) or a torch tensor ([H, W, c]).
    """
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str):
        return Image.open(image)
    elif isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            return Image.fromarray(image)
        else:
            image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = (image.detach().cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image)
    else:
        raise ValueError("Unsupported type of image")

class Img_refiner():
    def __init__(self,device="cuda"):
        self.pipeline = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
    
    def refine_image(self,init_image,prompt=None,strength=0.6,guidance_scale=4.0):
        if prompt is None:
            prompt = "empty. beautiful, attractive, pretty, good details, good anatomy"
        init_image = convert_PIL(init_image)
        w,h = init_image.size
        image = self.pipeline(prompt,height=h,width=w, image=init_image,strength=strength,guidance_scale=guidance_scale).images[0]
        image = np.array(image) / 255.0
        return image