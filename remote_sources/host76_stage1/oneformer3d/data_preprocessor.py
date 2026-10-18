# Copied from mmdet3d/models/data_preprocessors/data_preprocessor.py
from mmdet3d.models.data_preprocessors.data_preprocessor import \
    Det3DDataPreprocessor
from mmdet3d.registry import MODELS


@MODELS.register_module()
class Det3DDataPreprocessor_(Det3DDataPreprocessor):
    """
    We add only this 2 lines:
    if 'elastic_coords' in inputs:
        batch_inputs['elastic_coords'] = inputs['elastic_coords']
    """
    def simple_process(self, data, training=False):
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict

        if 'elastic_coords' in inputs:
            batch_inputs['elastic_coords'] = inputs['elastic_coords']

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if hasattr(self, 'boxtype2tensor') and self.boxtype2tensor:
                    from mmdet.models.utils.misc import \
                        samplelist_boxtype2tensor
                    samplelist_boxtype2tensor(data_samples)
                elif hasattr(self, 'boxlist2tensor') and self.boxlist2tensor:
                    from mmdet.models.utils.misc import \
                        samplelist_boxlist2tensor
                    samplelist_boxlist2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)

                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs
        
        def _normalize_frame_major(val):
            # Pseudo-collate keeps temporal fields produced by a single sample as
            # frame-major tuples: [(frame0_b0, ...), (frame1_b0, ...), ...].
            # Keep that layout. The previous transpose turned it into [B][T]
            # for tensors/paths and left cam_info as [T] dicts, which caused
            # frame>0 to reuse frame0 camera metadata and invalidated projection.
            if isinstance(val, list) and len(val) > 0:
                first = val[0]
                if isinstance(first, tuple):
                    return [list(frame_vals) for frame_vals in val]
                if isinstance(first, dict):
                    batch_size = 1
                    for vv in first.values():
                        if isinstance(vv, (list, tuple)):
                            batch_size = len(vv)
                            break
                    out = []
                    for frame_dict in val:
                        frame_batch = []
                        for b in range(batch_size):
                            one = {}
                            for k, vv in frame_dict.items():
                                if k in ('intrinsics', 'img_size_dino') and isinstance(vv, (list, tuple)):
                                    # Pseudo-collate turns [fx,fy,cx,cy] into
                                    # [(fx_b0,...), (fy_b0,...), ...]. Restore
                                    # the per-sample numeric vector instead of
                                    # taking only the first scalar.
                                    restored = []
                                    for elem in vv:
                                        if isinstance(elem, (list, tuple)) and len(elem) > b:
                                            restored.append(elem[b])
                                        else:
                                            restored.append(elem)
                                    one[k] = restored
                                elif isinstance(vv, (list, tuple)) and len(vv) > b:
                                    one[k] = vv[b]
                                else:
                                    one[k] = vv
                            frame_batch.append(one)
                        out.append(frame_batch)
                    return out
                if isinstance(first, list):
                    # Fallback for true sample-major [B][T] lists.
                    try:
                        n_frames = len(first)
                        return [[sample_frames[t] for sample_frames in val] for t in range(n_frames)]
                    except Exception:
                        return val
            return val

        for _key in ('img_paths', 'dino_feat_maps', 'cam_info', 'dino_cache_paths'):
            if _key in inputs:
                batch_inputs[_key] = _normalize_frame_major(inputs[_key])

        if 'img_path' in inputs:
            batch_inputs['img_path'] = inputs['img_path']
        return {'inputs': batch_inputs, 'data_samples': data_samples}