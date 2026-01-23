'''
render using frames in GS
inpaint with fooocus
'''
import os
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from ops.utils import *
from ops.sky import Sky_Seg_Tool
from ops.visual_check import Check
from ops.gs.train import GS_Train_Tool
from pipe.lvm_inpaint import Inpaint_Tool
from pipe.reconstruct import Reconstruct_Tool
from ops.trajs import _generate_trajectory
from ops.connect import Occlusion_Removal
from ops.gs.basic import Frame,Gaussian_Scene
from pipe.logger_config import logger
from ops.mcs import HackSD_MCS
from pipe.refine_mvdps import Refinement_Tool_MCS
from natsort import natsorted
        
class Pipeline():
    def __init__(self,cfg) -> None:
        self.device = 'cuda'
        self.cfg = cfg
        self.sky_value = cfg.model.sky.value
        self.sky_segor = Sky_Seg_Tool(cfg)
        self.rgb_inpaintor = Inpaint_Tool(cfg)
        self.reconstructor = Reconstruct_Tool(cfg)
        # temp
        self.removalor = Occlusion_Removal()
        self.checkor = Check()
        self.temp_img = None
        self.temp_result = []

    def _mkdir(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def _resize_input(self,fn):
        resize_long_edge = int(self.cfg.scene.input.resize_long_edge)
        print(f'[Preprocess...] Resize the long edge of input image to {resize_long_edge}.')
        spl = str.rfind(fn,'.')
        backup_fn = fn[:spl] + '.original' + fn[spl:]
        rgb = Image.open(fn)
        rgb.save(backup_fn) # back up original image 
        rgb = np.array(rgb)[:,:,:3]/255.
        H,W = rgb.shape[0:2]
        if H>W:
            W = int(W*resize_long_edge/H)
            H = resize_long_edge
        else:
            H = int(H*resize_long_edge/W)
            W = resize_long_edge
        rgb = cv2.resize(rgb,(W,H))
        pic = (rgb * 255.0).clip(0, 255)
        pic_save = Image.fromarray(pic.astype(np.uint8))
        pic_save.save(fn)

    def _determine_sky(self,out_rgb,out_dpt):
        sky = self.sky_segor(out_rgb)
        valid_dpt = out_dpt[~sky]
        _max = np.percentile(valid_dpt,95)
        self.sky_value = _max*1.2

    def _initialization(self,rgb):
        rgb = np.array(rgb)[:,:,:3]
        temp = Frame(rgb=rgb)
        self.initial_rgb = temp.rgb
        # conduct outpainting on rgb and change cu,cv
        outpaint_frame,_ = self.rgb_inpaintor(Frame(rgb=rgb),
                                                   outpaint_selections=self.outpaint_selections,
                                                   outpaint_extend_times=self.outpaint_extend_times,
                                                   bright_ness = False)
        # conduct reconstruction on outpaint results
        _,intrinsic,_ = self.reconstructor._ProDpt_(rgb) # estimate focal on input view
        metric_dpt,intrinsic,edge_msk = self.reconstructor._ProDpt_(outpaint_frame.rgb)
        self._determine_sky(outpaint_frame.rgb,metric_dpt)
        outpaint_frame.intrinsic = deepcopy(intrinsic)
        # split to input and outpaint areas
        input_frame = Frame(H=rgb.shape[0],
                            W=rgb.shape[1],
                            rgb=rgb,
                            intrinsic=deepcopy(intrinsic),
                            extrinsic=np.eye(4))
        input_frame.intrinsic[0,-1] = input_frame.W/2.
        input_frame.intrinsic[1,-1] = input_frame.H/2.
        # others
        input_area = ~outpaint_frame.inpaint
        input_edg = edge_msk[input_area].reshape(input_frame.H,input_frame.W)
        input_dpt = metric_dpt[input_area].reshape(input_frame.H,input_frame.W)
        sky = self.sky_segor(input_frame.rgb)
        input_frame.sky = sky
        input_dpt[sky] = self.sky_value
        input_frame.dpt = input_dpt
        input_frame.inpaint = np.ones_like(input_edg,bool)
        input_frame.inpaint_wo_edge = (~input_edg)
        input_frame.ideal_dpt = deepcopy(input_dpt)
        input_frame.prompt = outpaint_frame.prompt
        # outpaint frame
        sky = self.sky_segor(outpaint_frame.rgb)
        outpaint_frame.sky = sky
        metric_dpt[sky] = self.sky_value
        outpaint_frame.dpt = metric_dpt
        outpaint_frame.ideal_dpt = deepcopy(metric_dpt)
        outpaint_frame.inpaint = (outpaint_frame.inpaint)
        outpaint_frame.inpaint_wo_edge = (outpaint_frame.inpaint)&(~edge_msk)
        # temp visualization
        save_pic(outpaint_frame.rgb,self.coarse_interval_rgb_fn)
        # add init frame
        input_frame.keep = True
        outpaint_frame.keep = True
        self.scene._add_trainable_frame(input_frame,require_grad=True)
        self.scene._add_trainable_frame(outpaint_frame,require_grad=True)
        self.scene = GS_Train_Tool(self.scene,iters=100)(self.scene.frames)
    
    def _generate_traj(self):
        self.dense_trajs = _generate_trajectory(self.cfg,self.scene)
        
    def _pose_to_frame(self,pose,margin=32):
        extrinsic = np.linalg.inv(pose)
        H = self.scene.frames[0].H + margin
        W = self.scene.frames[0].W + margin
        prompt = self.scene.frames[-1].prompt
        intrinsic = deepcopy(self.scene.frames[0].intrinsic)
        intrinsic[0,-1], intrinsic[1,-1] = W/2, H/2
        frame = Frame(H=H,W=W,intrinsic=intrinsic,extrinsic=extrinsic,prompt=prompt)
        frame = self.scene._render_for_inpaint(frame)  
        frame.initial_rgb = self.initial_rgb
        return frame
      
    # def _next_frame(self,margin=32):
    #     # select the frame with largest holes but less than 60% 
    #     inpaint_area_ratio = []
    #     for pose in self.dense_trajs:
    #         temp_frame = self._pose_to_frame(pose,margin)
    #         inpaint_mask = temp_frame.inpaint 
    #         inpaint_area_ratio.append(np.mean(inpaint_mask))
    #     inpaint_area_ratio = np.array(inpaint_area_ratio)
    #     inpaint_area_ratio[inpaint_area_ratio > 0.6] = 0.
    #     # remove adjustancy frames
    #     for s in self.select_frames:
    #         inpaint_area_ratio[s] = 0.
    #         if s-1>-1:
    #             inpaint_area_ratio[s-1] = 0.
    #         if s+1<len(self.dense_trajs):
    #             inpaint_area_ratio[s+1] = 0.
    #     # select the largest ones
    #     select = np.argmax(inpaint_area_ratio)
    #     if inpaint_area_ratio[select] < 0.001: return None
    #     self.select_frames.append(select)
    #     pose = self.dense_trajs[select]
    #     frame = self._pose_to_frame(pose,margin)
    #     frame.prompt = self.scene_describtion[select]
    #     return frame  

    def _next_frame(self,margin=32, index = 0):
        if index < 0 or index >= len(self.cameras): return None
        pose = self.cameras[index]
        frame = self._pose_to_frame(pose, margin)
        # if index % 2 == 0:
        frame.prompt = self.scene_describtion[index]
        if self.dual_inpaint:
            frame.prompt_2 = self.scene_describtion_2[index]
        else:
            frame.prompt_2 = frame.prompt
        # elif index % 2 == 1:
        #     frame.prompt = self.scene_describtion_2[index]
        # else:

        # else:
        #     frame.prompt = None
        #     frame.prompt_2 = None

        inpaint_mask = frame.inpaint
        inpaint_area_ratio = np.mean(inpaint_mask)
        if inpaint_area_ratio < 0.001: return None

        return frame 

    def _inpaint_next_frame(self,margin=32, index = 0):
        frame = self._next_frame(margin, index = index)
        if frame is None: return None
        # inpaint rgb
        if self.refine:
            if 0 <= index < len(self.coarse_scene):
                frame, image = self.rgb_inpaintor(frame, refine_image = self.coarse_scene[index])
            else:
                frame, image = self.rgb_inpaintor(frame)
        
        if not self.refine:
            frame, image = self.rgb_inpaintor(frame, fix_prompt = self.fix_prompt)
            self.coarse_scene.append(image)
        # inpaint dpt
        connect_dpt,metric_dpt,_,edge_msk = self.reconstructor._Guide_ProDpt_(frame.rgb,frame.intrinsic,frame.dpt,~frame.inpaint)
        frame.dpt = connect_dpt
        frame = self.removalor(self.scene,frame)
        sky = self.sky_segor(frame.rgb)
        frame.sky = sky
        frame.dpt[sky] = self.sky_value
        frame.inpaint = (frame.inpaint) #& (~sky)
        frame.inpaint_wo_edge = (frame.inpaint) & (~edge_msk)
        # temp visualization
        save_pic(frame.rgb,self.coarse_interval_rgb_fn)
        self.temp_img = frame.rgb
        self.temp_result.append(frame.rgb)
        # determine target depth and normal
        frame.ideal_dpt = metric_dpt
        self.scene._add_trainable_frame(frame)
        return 0
    
    def rot_initial(self, rot_frame, inverse = False):
        t = np.linspace(0, 1, rot_frame)
        # rotation angles at each frame
        theta = -2 * np.pi * t * rot_frame 
        # try not to change y (up-down for floor and sky)
        pos = np.zeros((rot_frame,3))
        x = np.sin(theta)
        y = np.zeros_like(x)
        z = np.cos(theta)
        targets = np.vstack([x,y,z]).T
        if inverse:
            camera_ups = np.array([[0,-1.,0]]).repeat(rot_frame,axis=0)
        else:
            camera_ups = np.array([[0,1.,0]]).repeat(rot_frame,axis=0)
        # if inverse != None:
        #     camera_ups = np.array([[0,-1.,0]]).repeat(rot_frame,axis=0)
        # else:
        #     camera_ups = np.array([[0,1.,0]]).repeat(rot_frame,axis=0)
        # camera_triples
        cameras = []
        for i in range(rot_frame):
            cameras.append((pos[i],targets[i],camera_ups[i]))
        return cameras

    def trans_by_look_at(self, camera_triples):
        camera_poses = []
        for camera in camera_triples:
            pos,target,up = camera
            rotation_matrix = self.rot_by_look_at(pos,target,up)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3,  3] = pos
            camera_poses.append(transform_matrix[None])
        camera_poses = np.concatenate(camera_poses,axis=0)
        return camera_poses

    def rot_by_look_at(self, camera_position, target_position, camera_up):
        # look at direction
        direction = target_position - camera_position
        direction /= np.linalg.norm(direction)
        up = -camera_up # For the image origin is left-up: y is inverse
        up /= np.linalg.norm(up)
        # calculate rotation matrix
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        rotation_matrix = np.column_stack([right, up, direction])
        return rotation_matrix

    def _coarse_scene(self,rgb):
        self._initialization(rgb)
        self._generate_traj()
        self.select_frames = []
        for i in range(self.n_sample-2):
            print(f'Procecssing {i+2}/{self.n_sample} frame...')
            sign = self._inpaint_next_frame(index = i)
            if sign is None: break
            self.scene = GS_Train_Tool(self.scene,iters=self.opt_iters_per_frame)(self.scene.frames)

    def _MCS_Refinement(self):
        refiner = HackSD_MCS(device='cuda',use_lcm=True,denoise_steps=self.mcs_iterations,
                             sd_ckpt=self.cfg.model.optimize.sd,
                             lcm_ckpt=self.cfg.model.optimize.lcm)
        self.MVDPS = Refinement_Tool_MCS(self.scene,device='cuda',
                                         refiner=refiner,
                                         traj_type=self.traj_type,
                                         n_view=self.mcs_n_view,
                                         rect_w=self.mcs_rect_w,
                                         pre_blur=self.pre_blur,
                                         n_gsopt_iters=self.mcs_gsopt_per_frame)
        self.scene = self.MVDPS(temp_rgb_fn=self.refine_interval_rgb_fn)
        refiner.to('cpu')

    def get_image(self, image_path):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(original_image, (512, 512))
        original_image = original_image / 255.0
        return original_image

    def __call__(self, train_index = 0, dual_inpaint = True, coarse_scene = None, refine = False, fix_prompt = ''):
        self.dual_inpaint = dual_inpaint
        self.coarse_scene = coarse_scene
        self.refine = refine
        self.fix_prompt = fix_prompt

        rgb_fn = self.cfg.scene.input.rgb
        dir = rgb_fn[:str.rfind(rgb_fn,'/')]
        # temp_interval_image
        self.coarse_interval_rgb_fn = f'{dir}/temp.coarse.interval.png'
        self.refine_interval_rgb_fn = f'{dir}/temp.refine.interval.png'
        # coarse
        self.scene = Gaussian_Scene(self.cfg)
        # for trajectory genearation
        self.n_sample = self.cfg.scene.traj.n_sample
        self.traj_type = self.cfg.scene.traj.traj_type
        self.scene.traj_type = self.cfg.scene.traj.traj_type
        # for scene generation
        self.opt_iters_per_frame = self.cfg.scene.gaussian.opt_iters_per_frame
        self.outpaint_selections = self.cfg.scene.outpaint.outpaint_selections
        self.outpaint_extend_times = self.cfg.scene.outpaint.outpaint_extend_times
        # for scene refinement
        self.pre_blur = self.cfg.scene.mcs.blur
        self.mcs_n_view = self.cfg.scene.mcs.n_view
        self.mcs_rect_w = self.cfg.scene.mcs.rect_w
        self.mcs_iterations = self.cfg.scene.mcs.steps
        self.mcs_gsopt_per_frame = self.cfg.scene.mcs.gsopt_iters
        self.scene_describtion = []
        self.scene_describtion_2 = []
        self.cameras = self.rot_initial(20, inverse = True)
        self.cameras = self.trans_by_look_at(self.cameras)
        output_dir = ''

        with torch.no_grad():
            image_paths = os.listdir(output_dir)
            image_paths = natsorted(os.listdir(output_dir), reverse=True)
            for index, path in enumerate(image_paths):
                result = self.rgb_inpaintor.get_prompt((self.get_image(os.path.join(output_dir, path))))
                self.scene_describtion.append(result)
                print('Prompt:')
                print(result)
        
        if dual_inpaint:
            with torch.no_grad():
                image_paths = os.listdir(output_dir)
                image_paths = natsorted(os.listdir(output_dir), reverse=True)
                for index, path in enumerate(image_paths):
                    result = self.rgb_inpaintor.get_prompt((self.get_image(os.path.join(output_dir, path))), trans = 'left')
                    self.scene_describtion_2.append(result)
                    print('Prompt:')
                    print(result)

        # coarse scene
        self._resize_input(rgb_fn)
        rgb = Image.open(rgb_fn)
        self._coarse_scene(rgb)
        # for more vram
        self.sky_segor = None
        self.reconstructor = None
        self.rgb_inpaintor = None
        torch.cuda.empty_cache()
        # refinement
        
        if refine:
            self._MCS_Refinement()
            torch.save(self.scene,f'{dir}/scene.pth')
            self.checkor._render_video(self.scene,save_dir=f'{dir}/', postfix = str(train_index))
        else:
            self._MCS_Refinement()
            # torch.save(self.scene,f'{dir}/scene' + str(train_index) + '.pth')
            self.checkor._render_video(self.scene,save_dir=f'{dir}/', postfix = str(train_index))