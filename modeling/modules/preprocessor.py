import contextlib
import os
import os.path as osp
import sys
from typing import cast

import imageio.v3 as iio
import numpy as np
import torch


class Dust3rPipeline(object):
    def __init__(self, device: str | torch.device = "cuda"):
        submodule_path = osp.realpath(
            osp.join(osp.dirname(__file__), "../../third_party/dust3r/")
        )
        if submodule_path not in sys.path:
            sys.path.insert(0, submodule_path)
        try:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                from dust3r.cloud_opt import (  # type: ignore[import]
                    GlobalAlignerMode,
                    global_aligner,
                )
                from dust3r.image_pairs import make_pairs  # type: ignore[import]
                from dust3r.inference import inference  # type: ignore[import]
                from dust3r.model import AsymmetricCroCo3DStereo  # type: ignore[import]
                from dust3r.utils.image import load_images  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "Missing required submodule: 'dust3r'. Please ensure that all submodules are properly set up.\n\n"
                "To initialize them, run the following command in the project root:\n"
                "  git submodule update --init --recursive"
            )

        self.device = torch.device(device)
        self.model = AsymmetricCroCo3DStereo.from_pretrained(
            "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to(self.device)

        self._GlobalAlignerMode = GlobalAlignerMode
        self._global_aligner = global_aligner
        self._make_pairs = make_pairs
        self._inference = inference
        self._load_images = load_images

    def infer_cameras_and_points(
        self,
        img_paths: list[str],
        Ks: list[list] = None,
        c2ws: list[list] = None,
        batch_size: int = 16,
        schedule: str = "cosine",
        lr: float = 0.01,
        niter: int = 500,
        min_conf_thr: int = 3,
    ) -> tuple[
        list[np.ndarray], np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]
    ]:
        num_img = len(img_paths)
        if num_img == 1:
            print("Only one image found, duplicating it to create a stereo pair.")
            img_paths = img_paths * 2

        images = self._load_images(img_paths, size=512)
        pairs = self._make_pairs(
            images,
            scene_graph="complete",
            prefilter=None,
            symmetrize=True,
        )
        output = self._inference(pairs, self.model, self.device, batch_size=batch_size)

        ori_imgs = [iio.imread(p) for p in img_paths]
        ori_img_whs = np.array([img.shape[1::-1] for img in ori_imgs])
        img_whs = np.concatenate([image["true_shape"][:, ::-1] for image in images], 0)

        scene = self._global_aligner(
            output,
            device=self.device,
            mode=self._GlobalAlignerMode.PointCloudOptimizer,
            same_focals=True,
            optimize_pp=False,  # True,
            min_conf_thr=min_conf_thr,
        )

        # if Ks is not None:
        #     scene.preset_focal(
        #         torch.tensor([[K[0, 0], K[1, 1]] for K in Ks])
        #     )

        if c2ws is not None:
            scene.preset_pose(c2ws)

        _ = scene.compute_global_alignment(
            init="msp", niter=niter, schedule=schedule, lr=lr
        )

        imgs = cast(list, scene.imgs)
        Ks = scene.get_intrinsics().detach().cpu().numpy().copy()
        c2ws = scene.get_im_poses().detach().cpu().numpy()  # type: ignore
        pts3d = [x.detach().cpu().numpy() for x in scene.get_pts3d()]  # type: ignore
        if num_img > 1:
            masks = [x.detach().cpu().numpy() for x in scene.get_masks()]
            points = [p[m] for p, m in zip(pts3d, masks)]
            point_colors = [img[m] for img, m in zip(imgs, masks)]
        else:
            points = [p.reshape(-1, 3) for p in pts3d]
            point_colors = [img.reshape(-1, 3) for img in imgs]

        # Convert back to the original image size.
        imgs = ori_imgs
        Ks[:, :2, -1] *= ori_img_whs / img_whs
        Ks[:, :2, :2] *= (ori_img_whs / img_whs).mean(axis=1, keepdims=True)[..., None]

        return imgs, Ks, c2ws, points, point_colors
