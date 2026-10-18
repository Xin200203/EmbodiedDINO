# Adapted from mmdet3d/datasets/transforms/loading.py
import mmengine
import numpy as np
import torch
from typing import List, Optional, Union
import os, pdb, json

from mmdet3d.datasets.transforms import LoadAnnotations3D
from mmdet3d.datasets.transforms.loading import get
from mmdet3d.datasets.transforms.loading import NormalizePointsColor
from mmcv.transforms.base import BaseTransform
from mmcv.transforms import Compose, LoadImageFromFile

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import get_box_type
from mmdet3d.structures.points import BasePoints, get_points_type


@TRANSFORMS.register_module()
class LoadAnnotations3D_(LoadAnnotations3D):
    """Just add super point mask loading.
    
    Args:
        with_sp_mask_3d (bool): Whether to load super point maks. 
    """

    def __init__(self, with_sp_mask_3d, **kwargs):
        self.with_sp_mask_3d = with_sp_mask_3d
        super().__init__(**kwargs)

    def _load_sp_pts_3d(self, results):
        """Private function to load 3D superpoints mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        sp_pts_mask_path = results['super_pts_path']
        try:
            mask_bytes = get(
                sp_pts_mask_path, backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            sp_pts_mask = np.frombuffer(
                mask_bytes, dtype=np.int64).copy()
        except ConnectionError:
            mmengine.check_file_exist(sp_pts_mask_path)
            sp_pts_mask = np.fromfile(
                sp_pts_mask_path, dtype=np.int64)
        results['sp_pts_mask'] = sp_pts_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['sp_pts_mask'] = sp_pts_mask
            results['eval_ann_info']['lidar_idx'] = \
                sp_pts_mask_path.split("/")[-1][:-4]
        return results

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().transform(results)
        if self.with_sp_mask_3d:
            results = self._load_sp_pts_3d(results)
        return results


@TRANSFORMS.register_module()
class NormalizePointsColor_(NormalizePointsColor):
    """Just add color_std parameter.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
        color_std (list[float]): Std color of the point cloud.
            Default value is from SPFormer preprocessing.
    """

    def __init__(self, color_mean, color_std=127.5):
        self.color_mean = color_mean
        self.color_std = color_std

    def transform(self, input_dict):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
            Updated key and value are described below.
                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = input_dict['points']

        assert points.attribute_dims is not None and \
               'color' in points.attribute_dims.keys(), \
               'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                           points.color.new_tensor(self.color_mean)
        if self.color_std is not None:
            points.color = points.color / \
                points.color.new_tensor(self.color_std)
        input_dict['points'] = points
        return input_dict


@TRANSFORMS.register_module()
class LoadAdjacentDataFromFile(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normlize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normlize the elongation. This is
            usually used in Waymo dataset.Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 num_frames: int = 8,
                 num_sample: int = 20000,
                 max_frames: int = -1,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 with_bbox_3d=False,
                 with_label_3d=False,
                 with_mask_3d=True,
                 with_seg_3d=True,
                 with_sp_mask_3d=True,
                 with_rec=False,
                 cat_rec=False,
                 use_FF=False,
                 keep_img_info=False,
                 backend_args: Optional[dict] = None,
                 dataset_type = 'scannet200') -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        assert dataset_type in ['scannet', 'scannet200', 'scenenn', '3RScan']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.num_frames = num_frames
        self.num_sample = num_sample
        self.max_frames = max_frames
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.with_sp_mask_3d = with_sp_mask_3d
        self.with_rec = with_rec
        self.cat_rec = cat_rec
        self.use_FF = use_FF
        self.keep_img_info = keep_img_info
        self.backend_args = backend_args
        self.dataset_type = dataset_type
        
        self.loader = Compose([dict(type='LoadImageFromFile')])
        
        self.rotation_matrix = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ])
        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3,:3] = self.rotation_matrix

    def _load_points(self, pts_filenames):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:        
            points = [np.frombuffer(get(pts_filename, backend_args=self.backend_args), dtype=np.float32)
                for pts_filename in pts_filenames]
        except ConnectionError:
            if pts_filenames[0].endswith('.npy'):
                points = [np.load(pts_filename) for pts_filename in pts_filenames]
            else:
                points = [np.fromfile(pts_filename, dtype=np.float32) for pts_filename in pts_filenames]
        points = np.concatenate(points, axis=0)

        return points
    
    def _load_masks_3d(self, results, pts_instance_mask_paths):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        try:
            pts_instance_mask = [np.frombuffer(get(pts_instance_mask_path, backend_args=self.backend_args),
                 dtype=np.int64) for pts_instance_mask_path in pts_instance_mask_paths]
        except ConnectionError:
            pts_instance_mask = [np.fromfile(pts_instance_mask_path, dtype=np.int64)
                 for pts_instance_mask_path in pts_instance_mask_paths]
        pts_instance_mask = np.concatenate(pts_instance_mask, axis=0)

        results['pts_instance_mask'] = pts_instance_mask
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        return results

    def _load_semantic_seg_3d(self, results, pts_semantic_mask_paths):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        try:
            # add .copy() to fix read-only bug
            pts_semantic_mask = [np.frombuffer(get(pts_semantic_mask_path, backend_args=self.backend_args),
                 dtype=np.int64).copy() for pts_semantic_mask_path in pts_semantic_mask_paths]
        except ConnectionError:
            pts_semantic_mask = [np.fromfile(pts_semantic_mask_path, dtype=np.int64)
                 for pts_semantic_mask_path in pts_semantic_mask_paths]
        pts_semantic_mask = np.concatenate(pts_semantic_mask, axis=0)

        results['pts_semantic_mask'] = pts_semantic_mask # 160000
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results: # True
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
        return results
    
    def _load_sp_pts_3d(self, results, sp_pts_mask_paths):
        """Private function to load 3D superpoints mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        try:
            # add .copy() to fix read-only bug
            sp_pts_mask = [np.frombuffer(get(sp_pts_mask_path, backend_args=self.backend_args),
                 dtype=np.int64).copy() for sp_pts_mask_path in sp_pts_mask_paths]
        except ConnectionError:
            sp_pts_mask = [np.fromfile(sp_pts_mask_path, dtype=np.int64)
                 for sp_pts_mask_path in sp_pts_mask_paths]
        sp_pts_mask = np.array(sp_pts_mask)
        sp_pts_mask = np.concatenate(sp_pts_mask, axis=0)

        results['sp_pts_mask'] = sp_pts_mask # 160000
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['sp_pts_mask'] = sp_pts_mask
            results['eval_ann_info']['lidar_idx'] = \
                sp_pts_mask_paths[0].split("/")[-2]
        results['lidar_idx'] = sp_pts_mask_paths[0].split("/")[-2]
        return results
    
    def _load_rec_3d(self, results, pts_filenames):
        """Private function to load 3D superpoints mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        scene_name = pts_filenames[0].split('/')[-2]
        rec_pts_filename = 'data/' + self.dataset_type + '/points/' + scene_name + '.bin'
        rec_ins_path = 'data/' + self.dataset_type + '/instance_mask/' + scene_name + '.bin'
        rec_sem_path = 'data/' + self.dataset_type + '/semantic_mask/' + scene_name + '.bin'
        try:
            rec_pts = np.frombuffer(get(rec_pts_filename, backend_args=self.backend_args), dtype=np.float32)
            rec_ins = np.frombuffer(get(rec_ins_path, backend_args=self.backend_args), dtype=np.int64)
            rec_sem = np.frombuffer(get(rec_sem_path, backend_args=self.backend_args), dtype=np.int64).copy()
        except ConnectionError:
            rec_pts = np.fromfile(rec_pts_filename, dtype=np.float32)
            rec_ins = np.fromfile(rec_ins_path, dtype=np.int64)
            rec_sem = np.fromfile(rec_sem_path, dtype=np.int64)
        if self.dataset_type == 'scannet' or self.dataset_type == 'scannet200':
            segment_path = 'data/' + self.dataset_type + '/scans/' + scene_name + '/' + scene_name + '_vh_clean_2.0.010000.segs.json'
            segment_ids = np.array(json.load(open(segment_path))['segIndices'])
        if self.dataset_type == '3RScan':
            segment_path = 'data/' + self.dataset_type + '/3RScan/' + scene_name + '/' + 'mesh.refined.0.010000.segs.v2.json'
            segment_ids = np.array(json.load(open(segment_path))['segIndices'])
        if self.dataset_type == 'scenenn':
            segment_path = 'data/' + self.dataset_type + '/mesh_segs/' + scene_name + '.segs.json'
            segment_ids = np.array(json.load(open(segment_path))['segIndices'])

        rec_pts = rec_pts.reshape(-1, self.load_dim).copy()
        if self.dataset_type == 'scenenn':
            rec_pts.flags.writeable = True
            rec_pts[:,:3] = np.dot(self.rotation_matrix, rec_pts[:,:3].T).T
        results['rec_xyz'] = rec_pts
        results['rec_instance_mask'] = rec_ins
        results['rec_semantic_mask'] = rec_sem
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['rec_xyz'] = rec_pts
            results['eval_ann_info']['segment_ids'] = segment_ids
        return results

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        assert len(results['pts_paths']) == len(results['super_pts_paths']) \
            == len(results['pts_semantic_mask_paths']) == len(results['pts_instance_mask_paths'])
        pts_file_paths = results['pts_paths']
        pts_instance_mask_paths = results['pts_instance_mask_paths']
        pts_semantic_mask_paths = results['pts_semantic_mask_paths']
        sp_pts_mask_paths = results['super_pts_paths']
        if self.use_FF or self.keep_img_info:
            img_file_paths = list(results.get('img_paths', []))
            poses = list(results.get('poses', []))
        else:
            if 'img_paths' in results:
                del results['img_paths']
            if 'poses' in results:
                del results['poses']

        if self.num_frames > 0:
            begin_idx = np.random.randint(0, len(pts_file_paths))
            keep_view_idx = np.arange(begin_idx, begin_idx + self.num_frames)
            keep_view_idx %= len(pts_file_paths) # 这种应该是为了保证每次输入的帧数相同
            pts_file_paths = [pts_file_paths[idx] for idx in keep_view_idx]
            pts_instance_mask_paths = [pts_instance_mask_paths[idx] for idx in keep_view_idx]
            pts_semantic_mask_paths = [pts_semantic_mask_paths[idx] for idx in keep_view_idx]
            sp_pts_mask_paths = [sp_pts_mask_paths[idx] for idx in keep_view_idx]
            if self.use_FF or self.keep_img_info: # keep frame-aligned 2D metadata when using cached DINO
                img_file_paths = [img_file_paths[idx] for idx in keep_view_idx]
                poses = [poses[idx] for idx in keep_view_idx]

        if self.max_frames > 0 and len(pts_file_paths) > self.max_frames:
            choose_seq = np.floor(np.linspace(0, len(pts_file_paths) - 1, num=self.max_frames)).astype(np.int_)
            pts_file_paths = [pts_file_paths[idx] for idx in choose_seq]
            pts_instance_mask_paths = [pts_instance_mask_paths[idx] for idx in choose_seq]
            pts_semantic_mask_paths = [pts_semantic_mask_paths[idx] for idx in choose_seq]
            sp_pts_mask_paths = [sp_pts_mask_paths[idx] for idx in choose_seq]
            results['pts_paths'] = pts_file_paths
            results['pts_instance_mask_paths'] = pts_instance_mask_paths
            results['pts_semantic_mask_paths'] = pts_semantic_mask_paths
            results['super_pts_paths'] = sp_pts_mask_paths
            if self.use_FF or self.keep_img_info:
                img_file_paths = [img_file_paths[idx] for idx in choose_seq]
                results['img_paths'] = img_file_paths
                poses = [poses[idx] for idx in choose_seq]
                results['poses'] = poses

        points = self._load_points(pts_file_paths)
        points = points.reshape(-1, self.load_dim)

        if self.dataset_type == 'scenenn':
            points[:,:3] = np.dot(self.rotation_matrix, points[:,:3].T).T
        points = points[:, self.use_dim]
        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        if self.with_bbox_3d:
            raise NotImplementedError("bbox_3d is not needed for Online seg")
        
        if self.with_label_3d:
            raise NotImplementedError("label_3d is not needed for Online seg")

        if self.with_mask_3d: # True pts_instance_mask
            results = self._load_masks_3d(results, pts_instance_mask_paths) # pts_instance_mask (20000,)

        if self.with_seg_3d: # True pts_semantic_mask
            results = self._load_semantic_seg_3d(results, pts_semantic_mask_paths)

        if self.with_sp_mask_3d: # True sp_pts_mask (20000,) * N
            results = self._load_sp_pts_3d(results, sp_pts_mask_paths)
        
        if self.with_rec: # 这里应该是读取gt的mask，上面是读取生成的mask rec_xyz rec_instance_mask rec_semantic_mask
            results = self._load_rec_3d(results, pts_file_paths)
            if self.cat_rec: # 两种点云拼接
                points = np.concatenate([points, results['rec_xyz']], axis=0)
        
        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        
        if self.keep_img_info and not self.use_FF:
            results['img_paths'] = img_file_paths
            results['poses'] = poses
            if self.dataset_type == 'scenenn':
                results['poses'] = [(self.transform_matrix @ pose) for pose in poses]

        if self.use_FF:
            imgs = []
            for i in range(len(img_file_paths)):
                _results = dict(img_path=img_file_paths[i], img_prefix=None)
                _results = self.loader(_results)
                imgs.append(_results['img'])
            for key in _results.keys():
                if key not in ['img', 'img_prefix', 'img_path']:
                    results[key] = _results[key]
            results['img'] = imgs
            results['img_paths'] = img_file_paths
            results['poses'] = poses
            if self.dataset_type == 'scenenn':  
                results['poses'] = [(self.transform_matrix @ pose) for pose in poses]
        results['num_frames'] = len(pts_file_paths) if self.num_frames == -1 else self.num_frames
        results['num_sample'] = self.num_sample
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        return repr_str

@TRANSFORMS.register_module()
class LoadPointsFromFile_(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normlize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normlize the elongation. This is
            usually used in Waymo dataset.Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None,
                 dataset_type='scannet') -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args
        self.dataset_type = dataset_type
        
        self.rotation_matrix = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ])
        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3,:3] = self.rotation_matrix

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points.copy()

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results['lidar_points']['lidar_path']
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        points.flags.writeable = True
        if self.dataset_type == 'scenenn':
            points[:,:3] = np.dot(self.rotation_matrix, points[:,:3].T).T
        points = points[:, self.use_dim]
        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        return repr_str
  


@TRANSFORMS.register_module()
class LoadDINOFeatureFromCache(BaseTransform):
    """Load pre-extracted DINO feature maps for the MV frames selected by LoadAdjacentDataFromFile."""
    def __init__(self,
                 cache_root: str,
                 data_root: str = 'data/scannet200-mv_fast/',
                 key: str = 'dino_feat_maps',
                 dtype: str = 'float16',
                 strict: bool = True) -> None:
        self.cache_root = cache_root
        self.data_root = data_root.rstrip('/') + '/'
        self.key = key
        self.dtype = dtype
        self.strict = strict

    def _cache_path(self, img_path: str) -> str:
        p = str(img_path)
        if p.startswith(self.data_root):
            rel = p[len(self.data_root):]
        else:
            marker = 'scannet200-mv_fast/'
            rel = p.split(marker, 1)[1] if marker in p else p
        rel = os.path.splitext(rel)[0] + '.pt'
        return os.path.join(self.cache_root, rel)

    def transform(self, results: dict) -> dict:
        img_paths = results.get('img_paths', None)
        if img_paths is None:
            if self.strict:
                raise KeyError('LoadDINOFeatureFromCache requires img_paths; set LoadAdjacentDataFromFile.keep_img_info=True')
            return results
        feats, cache_paths = [], []
        for img_path in img_paths:
            cache_path = self._cache_path(img_path)
            cache_paths.append(cache_path)
            if not os.path.exists(cache_path):
                if self.strict:
                    raise FileNotFoundError(cache_path)
                feats.append(None)
                continue
            payload = torch.load(cache_path, map_location='cpu')
            feat = payload.get('feat') if isinstance(payload, dict) else payload
            if not torch.is_tensor(feat):
                raise TypeError(f'DINO cache payload has no tensor feat: {cache_path}')
            feat = feat.contiguous()
            if self.dtype in ('float16', 'fp16'):
                feat = feat.half()
            elif self.dtype in ('float32', 'fp32'):
                feat = feat.float()
            feats.append(feat)
        results[self.key] = feats
        results['dino_cache_paths'] = cache_paths
        return results


@TRANSFORMS.register_module()
class BuildCamInfoFromPoses(BaseTransform):
    """Build per-frame camera metadata for cached DINO projection."""
    def __init__(self,
                 dataset_type: str = 'scannet200',
                 target_size=(420, 560),
                 base_size=(480, 640),
                 max_depth: float = 20.0) -> None:
        self.dataset_type = dataset_type
        self.target_size = tuple(target_size)
        self.base_size = tuple(base_size)
        self.max_depth = float(max_depth)

    def _base_intrinsics(self):
        if self.dataset_type in ('scannet', 'scannet200'):
            return 577.870605, 577.870605, 319.5, 239.5
        if self.dataset_type == 'scenenn':
            return 544.47329, 544.47329, 320.0, 240.0
        return 577.870605, 577.870605, 319.5, 239.5

    def transform(self, results: dict) -> dict:
        poses = results.get('poses', None)
        img_paths = results.get('img_paths', None)
        if poses is None:
            raise KeyError('BuildCamInfoFromPoses requires poses; set LoadAdjacentDataFromFile.keep_img_info=True')
        fx, fy, cx, cy = self._base_intrinsics()
        base_h, base_w = self.base_size
        tgt_h, tgt_w = self.target_size
        sx, sy = float(tgt_w) / float(base_w), float(tgt_h) / float(base_h)
        intr = [fx * sx, fy * sy, cx * sx, cy * sy]
        cam_info = []
        for i, pose in enumerate(poses):
            pose_arr = np.asarray(pose, dtype=np.float32)
            cam_info.append(dict(
                pose=pose_arr,
                extrinsics=pose_arr,
                intrinsics=intr,
                img_size_dino=[int(tgt_h), int(tgt_w)],
                img_path=img_paths[i] if img_paths is not None and i < len(img_paths) else None,
                max_depth=self.max_depth,
            ))
        results['cam_info'] = cam_info
        return results
