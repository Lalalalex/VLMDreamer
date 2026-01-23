'''
render using frames in GS
inpaint with fooocus
'''
import torch
import numpy as np
from ops.llava import Llava
from ops.gs.basic import Frame
from ops.fooocus import Fooocus_Tool
import cv2
from pipe.logger_config import logger
from skimage import exposure
from skimage.transform import resize
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter
from piq import brisque
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
# from pipe.image_refine import Img_refiner

class Inpaint_Tool():
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self._load_model()
        self.index = 0
        # self.img_refiner = Img_refiner()

    def _load_model(self):
        self.fooocus = Fooocus_Tool(fooocus_ckpts=self.cfg.model.paint.fooocus.ckpts)
        # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting",
        #     torch_dtype=torch.float16,
        # ).to('cuda')
        # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-2-inpainting",
        #     torch_dtype=torch.float16
        # ).to('cuda')
        self.llava = Llava(device='cpu',llava_ckpt=self.cfg.model.vlm.llava.ckpt)

    def _llava_prompt(self,frame):
        prompt = '<image>\n \
                USER: Detaily imagine and describe the scene this image taken from? \
                \n ASSISTANT: This image is taken from a scene of ' 
        return prompt   

    def save_image(self, image, save_path):
        image_uint8 = (image * 255).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))

    def _llava_trans_prompt(self, trans):
        prompt = f'<image>\n \
               USER: Detaily imagine and describe what scene will be on the {trans} of image? \
               \nASSISTANT: The scene on the {trans} will be '
        return prompt
    
    def get_prompt(self, image, trans = 'right'):
        self.llava.model.to('cuda')
        query = self._llava_trans_prompt(trans)
        prompt = self.llava(image, query)
        split  = str.rfind(prompt,f'ASSISTANT: The scene on the {trans} will be ') + len(f'ASSISTANT: The scene on the {trans} will be ')
        prompt = prompt[split:]
        
        return prompt
    
    def light_enhance(self, mask_image, target_image, ref_image, alpha=0.3):
        target_lab = cv2.cvtColor((target_image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        ref_lab = cv2.cvtColor((ref_image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0

        matched_L = exposure.match_histograms(target_lab[..., 0], ref_lab[..., 0])
        adjusted_L = (1 - alpha) * target_lab[..., 0] + alpha * matched_L
        adjusted_L = np.clip(adjusted_L, 0, 1)

        if mask_image.shape[:2] != target_image.shape[:2]:
            mask_image = resize(mask_image, (target_image.shape[0], target_image.shape[1]), anti_aliasing=False)
            mask_image = (mask_image > 0.5).astype(np.float32)

        target_lab[..., 0] = adjusted_L
        matched_image = cv2.cvtColor((target_lab * 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

        return matched_image
    
    def histogram_matching(self, mask_image, target_image, ref_image, alpha=0.2):
        target_image = self.light_enhance(mask_image, target_image, ref_image)
        matched_image = exposure.match_histograms(target_image, ref_image, channel_axis=-1)
        result_image = (1 - alpha) * target_image + alpha * matched_image

        # mask_3d = np.repeat(mask_image[:, :, np.newaxis], 3, axis=2)
        # output_image = target_image.copy()
        # output_image[mask_3d] = result_image[mask_3d]


        return result_image
    
    def to_pil(self, image_float_rgb):
        image_uint8 = (np.clip(image_float_rgb, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(image_uint8)

    def mask_to_pil(self, mask_bool):
        mask_uint8 = (mask_bool.astype(np.uint8)) * 255
        return Image.fromarray(mask_uint8, mode="L")

    def pil_to_float_numpy(self, pil_img):
        return np.asarray(pil_img).astype(np.float32) / 255.0
    
    def compute_brisque_from_image(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Cannot read image.")
            return None
        
        image = cv2.resize(image, (512, 288))  # Resize to 512x512
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        gray_image = torch.tensor(gray_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        score = brisque(gray_image).item()
        
        print(f"BRISQUE score for {image_path}: {score}")
        return score

    def zero_middle_half_and_get_mask(self, frame_rgb):
        h, w, c = frame_rgb.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        left = w // 3
        right = (2 * w) // 3

        # frame_rgb[:, left:right, :] = 0
        mask[:, left:right] = 1

        return frame_rgb, mask

    def __call__(self, frame:Frame, outpaint_selections=[], outpaint_extend_times=0.0, bright_ness = True, fix_prompt = '', dual_inpaint = True, refine_image = None, index = 0):
        '''
        Must be Frame type
        '''
        # conduct reconstuction
        # ----------------------- LLaVA -----------------------
        if frame.prompt is None:
            print('Inpaint-Caption[1/3] Move llava.model to GPU...')
            self.llava.model.to('cuda')
            print('Inpaint-Caption[2/3] Llava inpainting instruction:')
            query  = self._llava_prompt(frame)
            prompt = self.llava(frame.rgb,query)
            prompt_2 = self.llava(frame.rgb,query)
            split  = str.rfind(prompt,'ASSISTANT: This image is taken from a scene of ') + len(f'ASSISTANT: This image is taken from a scene of ')
            prompt = prompt[split:]
            print(prompt) 
            print('Inpaint-Caption[3/3] Move llava.model to CPU...')
            self.llava.model.to('cpu')
            torch.cuda.empty_cache()
            frame.prompt = prompt

        elif frame.prompt == 'trans':
            print('Inpaint-Caption[1/3] Move llava.model to GPU...')
            print('Using trans prompt')
            self.llava.model.to('cuda')
            print('Inpaint-Caption[2/3] Llava inpainting instruction:')
            prompt = self.get_prompt(frame.temp)
            split  = str.rfind(prompt,f'ASSISTANT: The scene on the right will be ') + len(f'ASSISTANT: The scene on the right will be ')
            prompt = prompt[split:]
            prompt_2 = prompt
            print(prompt) 
            print('Inpaint-Caption[3/3] Move llava.model to CPU...')
            self.llava.model.to('cpu')
            torch.cuda.empty_cache()
            frame.prompt = prompt

        else:
            prompt = frame.prompt
            prompt_2 = frame.prompt_2
            print(f'Using pre-generated prompt: {prompt}')
        # --------------------- Fooocus ----------------------
        temp_path = ''
        print('Inpaint-Fooocus[1/2] Fooocus inpainting...')
        logger.info(prompt)
        image = frame.rgb

        original_size = frame.rgb.shape[:2][::-1]
        mask = np.zeros(image.shape[:2], dtype=bool) if len(outpaint_selections) > 0 else frame.inpaint
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        fooocus_result = image

        if refine_image is None:
            fooocus_result = self.fooocus(image_number=1,
                                prompt=prompt + ' 8K, no large circles, no cameras, no fisheye, bright.',
                                prompt_2=prompt_2 + ' 8K, no large circles, no cameras, no fisheye, bright.',
                                negative_prompt='Any fisheye, dark, any large circles, any blur, unrealism.',
                                outpaint_selections=outpaint_selections,
                                outpaint_extend_times=outpaint_extend_times,
                                origin_image=image,
                                mask_image=mask,
                                seed=self.cfg.scene.outpaint.seed)[0]
            if len(outpaint_selections) == 0:
                fooocus_result[~mask_3d] = image[~mask_3d]
            if bright_ness:
                # fooocus_result = self.increase_brightness(fooocus_result, mask)
                # fooocus_result = self.light_enhance(mask, fooocus_result, frame.initial_rgb)
                fooocus_result = self.histogram_matching(mask, fooocus_result, frame.initial_rgb)
        
            self.save_image(fooocus_result, temp_path)
            best_b = self.compute_brisque_from_image(temp_path)
            
            for i in range(3):
                new_result = self.fooocus(image_number=1,
                                prompt= prompt + ' 8K, no large circles, no cameras, no fisheye, bright.',
                                prompt_2=prompt_2 + ' 8K, no large circles, no cameras, no fisheye, bright.',
                                negative_prompt='Any fisheye, dark, any large circles, any blur, unrealism.',
                                outpaint_selections=outpaint_selections,
                                outpaint_extend_times=outpaint_extend_times,
                                origin_image=image,
                                mask_image=mask,
                                seed=self.cfg.scene.outpaint.seed + i)[0]
                if len(outpaint_selections) == 0:
                    fooocus_result[~mask_3d] = image[~mask_3d]
                if bright_ness:
                    # fooocus_result = self.increase_brightness(fooocus_result, mask)
                    # new_result = self.light_enhance(mask, new_result, frame.initial_rgb)
                    new_result = self.histogram_matching(mask, new_result, frame.initial_rgb)
                    # fooocus_result = self.laplacian_pyramid_blending(mask, new_result, frame.initial_rgb)
                self.save_image(new_result, temp_path)
                new_b = self.compute_brisque_from_image(temp_path)
                if new_b < best_b:
                    best_b = new_b
                    fooocus_result = new_result

        
        # if bright_ness:
        #     # fooocus_result = self.increase_brightness(fooocus_result, mask)
        #     # fooocus_result = self.light_enhance(mask, fooocus_result, frame.initial_rgb)
        #     fooocus_result = self.histogram_matching(mask, fooocus_result, frame.initial_rgb)
        # image = fooocus_result
        
        else:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            image[mask_3d] = refine_image[mask_3d]
        # if dual_inpaint:
            if len(outpaint_selections) == 0:
                _, mask_for_refine = self.zero_middle_half_and_get_mask(image)
                mask_inpaint = np.logical_and(mask, mask_for_refine)
                mask_3d = np.repeat(mask_inpaint[:, :, np.newaxis], 3, axis=2)
                inpaint_area_ratio = np.mean(mask_inpaint)
                if inpaint_area_ratio > 0.001:
                    fooocus_result = self.fooocus(image_number=1,
                                        prompt=prompt + ' 8K, no large circles, no cameras, no fisheye, bright.',
                                        prompt_2=prompt_2 + ' 8K, no large circles, no cameras, no fisheye, bright.',
                                        negative_prompt='Any fisheye, dark, any large circles, any blur, unrealism.',
                                        outpaint_selections=outpaint_selections,
                                        outpaint_extend_times=outpaint_extend_times,
                                        origin_image=image,
                                        mask_image=mask_inpaint,
                                        seed=self.cfg.scene.outpaint.seed)[0]
                    fooocus_result[~mask_3d] = image[~mask_3d]
                    # if bright_ness:
                    #     fooocus_result = self.histogram_matching(mask, fooocus_result, frame.initial_rgb, alpha = 0.3)
            
            # if bright_ness:
            #     # fooocus_result = self.increase_brightness(fooocus_result, mask)
            #     # fooocus_result = self.light_enhance(mask, fooocus_result, frame.initial_rgb)
            #     fooocus_result = self.histogram_matching(mask, fooocus_result, frame.initial_rgb)

        # self.save_image(fooocus_result, temp_path)
        # self.save_image(fooocus_result, '' +  + str(self.index) + '.jpg')


        # if refine_image:
        #     fooocus_result

        # for i in range(4):
        #     new_result = self.fooocus(image_number=1,
        #                     prompt= prompt + ' 8K, no large circles, no cameras, no fisheye, bright.',
        #                     negative_prompt='Any fisheye, dark, any large circles, any blur, unrealism.',
        #                     outpaint_selections=outpaint_selections,
        #                     outpaint_extend_times=outpaint_extend_times,
        #                     origin_image=image,
        #                     mask_image=mask,
        #                     seed=self.cfg.scene.outpaint.seed + i)[0]
        #     if bright_ness:
        #         # fooocus_result = self.increase_brightness(fooocus_result, mask)
        #         # new_result = self.light_enhance(mask, new_result, frame.initial_rgb)
        #         new_result = self.histogram_matching(mask, new_result, frame.initial_rgb)
        #         # fooocus_result = self.laplacian_pyramid_blending(mask, new_result, frame.initial_rgb)
        #     self.save_image(new_result, temp_path)
        #     new_b = self.compute_brisque_from_image(temp_path)
        #     if new_b < best_b:
        #         best_b = new_b
        #         fooocus_result = new_result
        
        # if refine_image:
        #     fooocus_result = self.img_refiner(fooocus_result)

        torch.cuda.empty_cache()

        self.save_image(fooocus_result, '' + str(self.index) + '.jpg')
        self.index = self.index + 1
        
        # reset the frame for outpainting
        if len(outpaint_selections) > 0.:
            assert len(outpaint_selections) == 4
            small_H, small_W = frame.rgb.shape[0:2]
            large_H, large_W = fooocus_result.shape[0:2]
            if frame.intrinsic is not None:
                # NO CHANGE TO FOCAL
                frame.intrinsic[0,-1] = large_W//2 
                frame.intrinsic[1,-1] = large_H//2 
            # begin sample pixel
            frame.H = large_H
            frame.W = large_W
            begin_H = (large_H-small_H)//2
            begin_W = (large_W-small_W)//2
            inpaint = np.ones_like(fooocus_result[...,0])
            inpaint[begin_H:(begin_H+small_H),begin_W:(begin_W+small_W)] *= 0.
            frame.inpaint = inpaint > 0.5
        frame.rgb = fooocus_result
        
        print('Inpaint-Fooocus[2/2] Assign Frame...')
        return frame, fooocus_result
    