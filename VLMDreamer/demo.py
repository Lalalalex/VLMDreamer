from pipe.cfgs import load_cfg
from pipe.c2f_recons import Pipeline
import gc
import torch

for i in range(20):
    for j in range(1):
        image_path = ''
        
        coarse_scene = []
        cfg = load_cfg(f'pipe/cfgs/basic.yaml')
        cfg.scene.input.rgb = image_path
        vistadream = Pipeline(cfg)
        vistadream(train_index = i, coarse_scene = coarse_scene, fix_prompt = prompt[j])

        del vistadream
        gc.collect()
        torch.cuda.empty_cache()  

        cfg = load_cfg(f'pipe/cfgs/basic.yaml')
        cfg.scene.input.rgb = image_path
        vistadream = Pipeline(cfg)
        vistadream(train_index = i, coarse_scene = coarse_scene, refine = True)

        del vistadream
        gc.collect()
        torch.cuda.empty_cache()  