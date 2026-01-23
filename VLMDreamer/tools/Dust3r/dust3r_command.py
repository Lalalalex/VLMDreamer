import copy
import numpy as np
import open3d as o3d
from PIL import Image
from dataclasses import dataclass
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.inference import inference, load_model
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

@dataclass
class Dust3rOutput():
    rgb: np.array
    dpt: np.array
    conf: np.array
    intrinsic: np.array   
    extrinsic: np.array   
    

def visual_pcd(xyz, color=None, normal = True):
    if hasattr(xyz,'ndim'):
        xyz = xyz.reshape(-1,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
    else: pcd = xyz
    if color is not None:
        color = color.reshape(-1,3)
        pcd.colors = o3d.utility.Vector3dVector(color)
    if normal:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.2, 20))
    o3d.visualization.draw_geometries([pcd])

def points_to_hpoints(points):
    n,_=points.shape
    return np.concatenate([points,np.ones([n,1])],1)

def hpoints_to_points(hpoints):
    return hpoints[:,:-1]/hpoints[:,-1:]

def transform_points(pts,transform):
    h,w=transform.shape
    if h==3 and w==3:
        return pts @ transform.T
    if h==3 and w==4:
        return pts @ transform[:,:3].T + transform[:,3:].T
    elif h==4 and w==4:
        return hpoints_to_points(points_to_hpoints(pts) @ transform.T)
    else: raise NotImplementedError

def dpt2xyz(dpt,intrinsic):
    # get grid
    height, width = dpt.shape[0:2]
    grid_u = np.arange(width)[None,:].repeat(height,axis=0)
    grid_v = np.arange(height)[:,None].repeat(width,axis=1)
    grid = np.concatenate([grid_u[:,:,None],grid_v[:,:,None],np.ones_like(grid_v)[:,:,None]],axis=-1)
    uvz = grid * dpt[:,:,None]
    # inv intrinsic
    inv_intrinsic = np.linalg.inv(intrinsic)
    xyz = np.einsum(f'ab,hwb->hwa',inv_intrinsic,uvz)
    return xyz

class Dust3r():
    def __init__(self,
                 ckpt = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
                 device = 'cuda') -> None:
        self.ckpt = ckpt
        self.device = device
        self._load_model()
    
    def _load_model(self):
        self.model = load_model(self.ckpt, self.device).eval()

    def __call__(self, filelist, image_size=512,
                 schedule='linear', niter=300, scenegraph_type='complete', winsize=1, refid=0):
        """
        from a list of images, run dust3r inference, global aligner.
        then run get_3D_model_from_scene
        """
        img = np.array(Image.open(filelist[0]))
        h_origin,w_origin=img.shape[0:2]
        
        imgs = load_images(filelist, size=image_size)
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
        if scenegraph_type == "swin":
            scenegraph_type = scenegraph_type + "-" + str(winsize)
        elif scenegraph_type == "oneref":
            scenegraph_type = scenegraph_type + "-" + str(refid)

        pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, self.model, self.device, batch_size=1)

        mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)
        lr = 0.01

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

        # get optimized values from scene
        rgbs = scene.imgs
        focals = scene.get_focals().cpu()
        dpts = to_numpy(scene.get_depthmaps())
        cams2world = scene.get_im_poses().detach().cpu().numpy()
        confs = to_numpy([c for c in scene.im_conf])
        
        frames = []
        for i in range(len(rgbs)):
            rgb = rgbs[i]
            f = focals[i]
            H,W = rgb.shape[0:2]
            intrinsic = np.eye(3)
            intrinsic[0,0],intrinsic[1,1],intrinsic[0,-1],intrinsic[1,-1] = f,f,W/2,H/2
            extrinsic = np.linalg.inv(cams2world[i])
            rgb = rgbs[i]
            dpt = dpts[i]
            conf = confs[i]
            
            frame = Dust3rOutput(rgb=rgb,dpt=dpt,conf=conf,intrinsic=intrinsic,extrinsic=extrinsic)
            frames.append(frame)
        return frames


if __name__=='__main__':
    from PIL import Image
    runner = Dust3r()
    imgs = []
    imgs.append('/mnt/proj/Omni3D-Clean/trainset/HyperSim/train/ai_001_001_cam_00/images/frame_0000.png')
    imgs.append('/mnt/proj/Omni3D-Clean/trainset/HyperSim/train/ai_001_001_cam_00/images/frame_0001.png')
    # imgs.append('/mnt/proj/Omni3D-Clean/trainset/HyperSim/train/ai_001_001_cam_00/images/frame_0002.png')
    output = runner(imgs)
    