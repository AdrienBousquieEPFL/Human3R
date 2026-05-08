import os
import torch
import numpy as np
import pickle
from eval.global_human.data_utils import *

# Define the merged dataset metadata dictionary

def create_emdb(split):
    return {
        "img_path": "/work/courses/digital_human/team8/EMDB_dataset",
        "dir_path_func": lambda img_path, seq: os.path.join(f"{img_path}/{seq}/images"),
        "seq_list": None,
        "full_seq": True,
        "mask_path_func": lambda filelist: [],
        "split": split,
        "subsample": 1,
        "get_view_func": lambda inputs: load_view_emdb(*inputs),
        "get_seq_func": lambda img_path, split, annots: get_seq_emdb(img_path, split, annots),
        "get_annot_func": lambda img_path, split: get_annot_emdb(img_path, split),
        "is_global": lambda split: {1: False, 2: True}[split],
    }

dataset_metadata = {
    "bedlam": {
        "img_path": "/path/to/processed_bedlam",
        "dir_path_func": lambda img_path, seq: os.path.join(f"{img_path}/Test/{seq}/rgb"),
        "seq_list": None,
        "full_seq": True,
        "mask_path_func": lambda filelist: [f.replace("/rgb/", "/mask/") for f in filelist],
        "split": "Test",
        "subsample": 25, # 25 is used in Multi-HMR's code
        "get_view_func": lambda inputs: load_view_bedlam(*inputs[:2]),
        "get_seq_func": None,
        "is_global": lambda split: False,
    },
    "3dpw": {
        "img_path": "/path/to/3DPW",
        "dir_path_func": lambda img_path, seq: os.path.join(f"{img_path}/imageFiles/{seq}"),
        "seq_list": None,
        "full_seq": True,
        "mask_path_func": lambda filelist: [],
        "split": "test",
        "subsample": 1,
        "get_view_func": lambda inputs: load_view_3dpw(*inputs[:3]),
        "get_seq_func": lambda img_path, split, annots: get_seq_3dpw(img_path, split, annots),
        "get_annot_func": lambda img_path, split: get_annot(img_path, split, "3dpw"),
        "is_global": lambda split: False,
    },
    "emdb1": create_emdb(1),
    "emdb2": create_emdb(2),
    "rich": {
        "img_path": "/path/to/RICH",
        "dir_path_func": lambda img_path, seq: os.path.join(f"{img_path}/{seq}"),
        "seq_list": None,
        "full_seq": True,
        "mask_path_func": lambda filelist: [],
        "split": "test",
        "subsample": 1,
        "get_view_func": lambda inputs: load_view_rich(*inputs),
        "get_seq_func": lambda img_path, split, annots: get_seq_rich(img_path, split, annots),
        "get_annot_func": lambda img_path, split: get_annot_rich(),
        "is_global": lambda split: True,
    }
}


def get_annot(img_path, split, dataset):
    annot_file = os.path.join(f"eval/global_human/annots/{dataset}_{split}.pkl")
    with open(annot_file, 'rb') as f:
        annots = pickle.load(f)
    return annots

def get_annot_emdb(img_path, split):
    annots = {}
    for pkl_name in EMDB_LIST[split]:
        data = load_pkl(os.path.join(img_path, pkl_name))
        annots[data["name"]] = data
    return annots

def get_annot_rich():
    """
    For annotations of RICH dataset, 
    please download from https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md#inputs--outputs
    """
    annot_path = os.path.join(os.path.dirname(__file__), "annots/RICH")
    annots = torch.load(os.path.join(annot_path, "hmr4d_support/rich_test_labels.pt"))
    cam_params = torch.load(os.path.join(annot_path, "resource/cam2params.pt"))

    for vid in list(annots.keys()):
        _, sname, cname = vid.split("/")
        scene = sname.split("_")[0]
        cid = int(cname.split("_")[1])
        cam_key = f"{scene}_{cid}"
        annots[vid]["T_w2c"], annots[vid]["K"] = cam_params[cam_key]
    
    return annots

def get_seq_emdb(img_path, split, annots=None):
    if annots is None:
        annots = get_annot_emdb(img_path, split)

    seq_list = list(annots.keys())
    seq_to_images = {}
    for seq in seq_list:
        mask = annots[seq]["mask"]
        seq_images_dir = os.path.join(img_path, seq, "images")
        all_images = np.array(sorted(os.listdir(seq_images_dir)))
        seq_to_images[seq] = all_images[mask].tolist()

    return seq_list, seq_to_images

def get_seq_rich(img_path, split, annots=None):
    if annots is None:
        annots = get_annot_rich()

    seq_list = list(annots.keys())

    seq_to_images = {}
    for seq in seq_list:
        mask = annots[seq]["frame_id"]
        seq_images_dir = os.path.join(img_path, seq)
        all_images = np.array(sorted(os.listdir(seq_images_dir)))
        seq_to_images[seq] = all_images[mask].tolist()

    return seq_list, seq_to_images

def get_seq_3dpw(img_path, split, annots=None):
    if annots is None:
        annots = get_annot(img_path, split, "3dpw")

    imagenames = sorted(annots.keys())

    seq_to_images = {}
    for imgname in imagenames:
        seq, image = imgname.split('/')
        seq_to_images.setdefault(seq, []).append(image)

    return list(seq_to_images.keys()), seq_to_images

def load_view_bedlam(img_paths, images):
    max_humans = 10
    smpl_key2shape = {
        'smplx_root_pose': (1, 3), 
        'smplx_body_pose': (21, 3), 
        'smplx_jaw_pose': (1, 3), 
        'smplx_leye_pose': (1, 3), 
        'smplx_reye_pose': (1, 3), 
        'smplx_left_hand_pose': (15, 3), 
        'smplx_right_hand_pose': (15, 3), 
        'smplx_shape': (11,), 
        'smplx_transl': (3,), 
        'smplx_gender_id': (),
        }
    for img_path, image in zip(img_paths, images):
        # load camera
        cam_path = img_path.replace("/rgb/", "/cam/").replace(".png", ".npz")
        cam = np.load(cam_path)
        camera_pose = cam["pose"]
        intrinsics = cam["intrinsics"]

        # calculate scale factor
        fy_scale, fx_scale = (image['true_shape'] / image['ori_shape'])[0]

        # update intrinsics
        intrinsics[0, 0] *= fx_scale  # fx
        intrinsics[1, 1] *= fy_scale  # fy
        intrinsics[0, 2] *= fx_scale  # cx
        intrinsics[1, 2] *= fy_scale  # cy

        # load smpl
        annot_file = img_path.replace("/rgb/", "/smpl/").replace(".png", ".pkl")
        annots = []
        smpl_mask = np.zeros(max_humans, dtype=np.bool_)

        if os.path.isfile(annot_file):
            with open(annot_file, 'rb') as f:
                annots = pickle.load(f)
            humans = [hum for hum in annots]
            # humans = [hum for hum in annots if hum['smplx_transl'][-1] > 0.01] # the person should be in front of the camera
            if len(humans) > 0:
                smpl_mask[:len(humans)] = 1.
                l_dist = [hum['smplx_transl'][-1] for hum in humans]
                indexed_lst = list(enumerate(l_dist))
                sorted_indexed = sorted(indexed_lst, key=lambda x: x[1], reverse=False)
                sorted_indices = [index for index, _ in sorted_indexed]
                annots = [humans[h_idx] for h_idx in sorted_indices]

                # Update smplx_gender - 0=neutral - 1=male - 2=female - kids?
                for hum in annots:
                    hum['smplx_gender_id'] = np.asarray({'neutral': 0}[hum['smplx_gender']])

        smpl_dict = {}
        for k, shape in smpl_key2shape.items():
            smpl_dict[k] = np.zeros((max_humans, *shape), dtype=np.float32)
            if len(humans) > 0:
                for h in range(len(humans)):
                    smpl_dict[k][h] = annots[h][k].astype(np.float32)

        image.update({
            'camera_pose': camera_pose.astype(np.float32),
            'intrinsics': intrinsics.astype(np.float32),
            'smpl_mask': smpl_mask,
            **smpl_dict,
        })

    return images

def load_view_3dpw(img_paths, images, annots):
    max_humans = 2
    smpl_key2shape = {
        'smpl_root_pose': (1, 3), 
        'smpl_body_pose': (23, 3), 
        'smpl_shape': (10,), 
        'smpl_transl': (3,), 
        'smpl_gender_id': (),
        }

    for img_path, image in zip(img_paths, images):
        img_key = '/'.join(img_path.split('/')[-2:])

        # load camera
        T_w2c = annots[img_key]["cam_poses"]
        camera_pose = np.linalg.inv(T_w2c) # T_c2w
        focal = annots[img_key]["focal"]
        princpt = annots[img_key]["princpt"]
        intrinsics = np.array([
            [focal[0], 0, princpt[0]],
            [0, focal[1], princpt[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # calculate scale factor
        fy_scale, fx_scale = (image['true_shape'] / image['ori_shape'])[0]

        # update intrinsics
        intrinsics[0, 0] *= fx_scale  # fx
        intrinsics[1, 1] *= fy_scale  # fy
        intrinsics[0, 2] *= fx_scale  # cx
        intrinsics[1, 2] *= fy_scale  # cy

        # load smpl
        humans = annots[img_key]["humans"]
        smpl_mask = np.zeros(max_humans, dtype=np.bool_)

        if len(humans) > 0:
            smpl_mask[:len(humans)] = 1.
            l_dist = [hum['smpl_transl'][-1] for hum in humans]
            indexed_lst = list(enumerate(l_dist))
            sorted_indexed = sorted(indexed_lst, key=lambda x: x[1], reverse=False)
            sorted_indices = [index for index, _ in sorted_indexed]
            annots_humans = [humans[h_idx] for h_idx in sorted_indices]

            # Update smplx_gender - 0=neutral - 1=male - 2=female - kids?
            for hum in annots_humans:
                hum['smpl_gender_id'] = np.asarray({'male': 1, 'female': 2}[hum['smpl_gender']])

        smpl_dict = {}
        for k, shape in smpl_key2shape.items():
            smpl_dict[k] = np.zeros((max_humans, *shape), dtype=np.float32)
            if len(humans) > 0:
                for h in range(len(humans)):
                    smpl_dict[k][h] = annots_humans[h][k].astype(np.float32)

        image.update({
            'camera_pose': camera_pose.astype(np.float32),
            'T_w2c': T_w2c.astype(np.float32),
            'intrinsics': intrinsics.astype(np.float32),
            'smpl_mask': smpl_mask,
            **smpl_dict,
        })

    return images

def load_view_emdb(img_paths, images, annots, indices):
    seq = '/'.join(img_paths[0].split('/')[-4:-2])
    gender = annots[seq]['gender']
    masks = annots[seq]['mask']

    # load camera
    T_w2c = annots[seq]['T_w2c'][masks] # (n_frame,4,4)
    camera_poses = np.linalg.inv(T_w2c) # T_c2w: (n_frame,4,4)
    intrinsics = annots[seq]['K_fullimg'].copy() # (3,3)

    # update intrinsics
    fy_scale, fx_scale = (
        images[0]['true_shape'] / images[0]['ori_shape'])[0]
    intrinsics[0, 0] *= fx_scale  # fx
    intrinsics[1, 1] *= fy_scale  # fy
    intrinsics[0, 2] *= fx_scale  # cx
    intrinsics[1, 2] *= fy_scale  # cy

    # load smpl
    smpl_params = annots[seq]['smpl_params']    # world space

    max_humans = 1
    smpl_key2shape = {
        'smpl_root_pose_w': (1, 3), 
        'smpl_body_pose': (23, 3), 
        'smpl_shape': (10,), 
        'smpl_transl_w': (3,), 
        }

    for img_id, image in zip(indices, images):
        smpl_mask = np.ones(max_humans, dtype=np.bool_)
        smpl_dict = {}
        for k, shape in smpl_key2shape.items():
            smpl_dict[k] = np.zeros((max_humans, *shape), dtype=np.float32)
            smpl_dict[k][0] = smpl_params[k][masks][img_id].reshape(*shape).astype(np.float32)

        image.update({
            'T_w2c': T_w2c[img_id].astype(np.float32),
            'camera_pose': camera_poses[img_id].astype(np.float32),
            'intrinsics': intrinsics.astype(np.float32),
            'smpl_mask': smpl_mask,
            'smpl_gender_id': np.asarray({'male': 1, 'female': 2}[gender]),
            **smpl_dict,
        })

    return images

def load_view_rich(img_paths, images, annots, indices):
    seq = '/'.join(img_paths[0].split('/')[-4:-1])
    gender = annots[seq]['gender']

    # load camera
    T_w2c = annots[seq]['T_w2c'].numpy() # (4,4)
    camera_poses = np.linalg.inv(T_w2c) # T_c2w: (4,4)
    intrinsics = annots[seq]['K'].numpy().copy() # (3,3)

    # update intrinsics
    fy_scale, fx_scale = (
        images[0]['true_shape'] / images[0]['ori_shape'])[0]
    intrinsics[0, 0] *= fx_scale  # fx
    intrinsics[1, 1] *= fy_scale  # fy
    intrinsics[0, 2] *= fx_scale  # cx
    intrinsics[1, 2] *= fy_scale  # cy

    # load smpl
    smpl_params = annots[seq]['gt_smplx_params']    # world space

    max_humans = 1
    smpl_key2shape = {
        'global_orient': (1, 3), 
        'body_pose': (21, 3), 
        'betas': (10,), 
        'transl': (3,), 
        }

    for img_id, image in zip(indices, images):
        smpl_mask = np.ones(max_humans, dtype=np.bool_)
        smpl_dict = {}
        for k, shape in smpl_key2shape.items():
            smpl_dict['smplx_'+k] = np.zeros((max_humans, *shape), dtype=np.float32)
            smpl_dict['smplx_'+k][0] = smpl_params[k][img_id].reshape(*shape).numpy().astype(np.float32)

        image.update({
            'T_w2c': T_w2c.astype(np.float32),
            'camera_pose': camera_poses.astype(np.float32),
            'intrinsics': intrinsics.astype(np.float32),
            'smpl_mask': smpl_mask,
            'smplx_gender_id': np.asarray({'male': 1, 'female': 2}[gender]),
            **smpl_dict,
        })

    return images
