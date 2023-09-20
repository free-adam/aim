import math, torch
import py3d_tools as p3dT
import disco_xform_utils as dxf
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
import cv2 
import torchvision.transforms as T
#from infer import InferenceHelper

def init_midas_depth_model(midas_model_path, device):
    print(midas_model_path)
    midas_model = None
    net_w = None
    net_h = None
    normalization = None

    print("initialize MIDAS depth model")


    midas_model = DPTDepthModel(
        path=midas_model_path,
        backbone="vitl16_384",
        non_negative=True,
    )
    net_w, net_h = 384, 384
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    

    midas_transform = T.Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method='minimal',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    midas_model.eval()

    midas_model = midas_model.to(memory_format=torch.channels_last)
    midas_model = midas_model.half()

    midas_model.to(device)

    return midas_model, midas_transform

def do(src_img_path, camera_motion, midas_model, midas_transform, param, device):
    TRANSLATION_SCALE = 1.0/200.0
    tx, ty, tz, rx, ry, rz = camera_motion
    t_xyz = [-tx*TRANSLATION_SCALE, ty*TRANSLATION_SCALE, -tz*TRANSLATION_SCALE]
    r_xyz_degrees = [rx, ry, rz]
    r_xyz = [math.radians(r_xyz_degrees[0]), math.radians(r_xyz_degrees[1]), math.radians(r_xyz_degrees[2])]
    r_mat = p3dT.euler_angles_to_matrix(torch.tensor(r_xyz, device=device), "XYZ").unsqueeze(0)
    target_img = dxf.transform_image_3d(src_img_path, 
                                        midas_model, 
                                        midas_transform, 
                                        device,
                                        r_mat, 
                                        t_xyz, 
                                        int(param['3d_warping']['near_plane']), 
                                        int(param['3d_warping']['far_plane']),
                                        int(param['3d_warping']['fov']),
                                        padding_mode=param['3d_warping']['padding_mode'],
                                        sampling_mode=param['3d_warping']['sampling_mode'],
                                        midas_weight=float(param['3d_warping']['midas_weight']))
    return target_img