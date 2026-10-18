# excerpt: DGI init/merge/predict diagnostics from oneformer3d/mixformer3d.py
                 merge_sp_masks = False,
                 replace_bn_with_ln=False,
                 debug_mode=False,
                 dino_enable=False,
                 dino_require=False,
                 dino_debug=False,
                 dino_pose_mode='auto',
                 dino_valid_rate_warn_thr=0.90,
                 dgi_enable=False,
                 dgi_dino_thr=0.82,
                 dgi_weight=0.50,
                 dgi_cluster_thr=0.50,
                 dgi_debug=False,
                 use_query_merge_fusion=True
                 ):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if memory is not None:
            self.memory = MODELS.build(memory)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        if merge_head is not None:
            self.merge_head = MODELS.build(merge_head)
        if merge_criterion is not None:
            self.merge_criterion = MODELS.build(merge_criterion)
        self.use_query_merge_fusion = bool(use_query_merge_fusion)
        if getattr(self.decoder, '_dual_path', False) and self.use_query_merge_fusion:
            self.query_merge_fusion = nn.Linear(2 * self.decoder.d_model, self.decoder.d_model, bias=False)
            with torch.no_grad():
                self.query_merge_fusion.weight[:, :self.decoder.d_model].copy_(torch.eye(self.decoder.d_model))
                self.query_merge_fusion.weight[:, self.decoder.d_model:].copy_(torch.eye(self.decoder.d_model))
        self.criterion = MODELS.build(criterion)
        self.decoder_online = decoder['temporal_attn']
        self.use_bbox = decoder['bbox_flag']
        self.sem_len = decoder['num_semantic_classes'] + 1 # 201
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.map_to_rec_pcd = map_to_rec_pcd
        # self.use_query_memory = decoder['use_query_memory']
        self.use_query_memory = use_query_memory
        if self.use_query_memory:
            self.muti_scale_query = MultiScaleQuery()
            self.query_memory = None
            self.pos_memory = None
            self.query_memory_relu = nn.ReLU()
            self.muti_scale_query.init_weights()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.muti_scale_self_attn = MultiScaleQuery()
            self.self_attn_relu = nn.ReLU()
            self.muti_scale_self_attn.init_weights()
        self.use_noise = use_noise
        if self.use_noise:
            self.noise_p = noise_p
            self.noise_k = noise_k
        self.use_temporal_loss = use_temporal_loss
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_mask_memory = None
            self.before_sp_xyz = None
            self.before_query_ids = None
        self.use_decouple = use_decouple
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_query_boxes = None
        self.use_mot = use_mot
        if self.use_mot:
            self.mot_type = mot_type
            if mot_type == 'dq_track':
                self.use_relative_asso = use_relative_asso
                if self.use_relative_asso:
                    self.embed_trans2 = nn.Linear(256, 1)
                    self.heatmap_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
                self.use_refine = use_refine
                # self.asso_loss_weight = asso_loss_weight
                self.iou_calculator = AxisAlignedBboxOverlaps3D()
                self.matcher = TASK_UTILS.build(matcher)
                self.tracklet_trans = DQ_FFN(d_model=256, d_ffn=256, dropout=0)
                self.detector_trans = DQ_FFN(d_model=256, d_ffn=256, dropout=0)
                self.box_trans = nn.Sequential(
                    nn.Linear(6, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
                # query_trans = {'with_att': True, 'with_pos': True, 'min_channels': 256, 'drop_rate': 0.0}
                query_trans = asso_config['query_trans'] if asso_config is not None else {'with_att': True, 'with_pos': True, 'min_channels': 256, 'drop_rate': 0.0}
                self.query_inter = QueryInteractionX(in_channels=256, mid_channels=256, **query_trans)
                # self.update_type = asso_config['update_type'] if asso_config is not None else 'ema'
                self.asso_config = EasyDict(asso_config)
                
                self.rel_dist_embed = nn.Sequential(
                    nn.Linear(1, 256),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
                self.embed_trans = nn.Linear(256, 1)
                loss_asso = {'use_sigmoid': False, 'loss_weight': 1.0}
                from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
                self.loss_asso = CrossEntropyLoss(**loss_asso)
                self.ema_decay_rate = 0.5
                # self.train_asso_only = train_asso_only
            else:
                raise NotImplementedError(f"mot_type {mot_type} is not supported")

        self.use_one2many = use_one2many
        if self.use_one2many:
            self.one2many_loss_weight = 0.5
            self.criterion_one2many = MODELS.build(criterion_one2many[0])
        self.reweight_dict = reweight_dict
        self.merge_sp_masks = merge_sp_masks
        if self.merge_sp_masks:
            self.acc_dict = {}
            self.merge_box_trans = nn.Sequential(
                nn.Linear(6, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                DQ_FFN(d_model=256, d_ffn=256, dropout=0))
            self.merge_dist_embed = nn.Sequential(
                    nn.Linear(1, 256),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
            self.merge_embed_trans = nn.Linear(256, 1)
            self.merge_heatmap_loss = nn.BCEWithLogitsLoss(reduction='mean')
            self.merge_iou_calculator = AxisAlignedBboxOverlaps3D()
            self.fuse_linear = nn.Sequential(
                nn.Linear(512, 256, bias=False),
                nn.ReLU(inplace=True),
                nn.LayerNorm(256)
            )
            self.merge_fusion = MergeFusion(256)
            query_trans = {'with_att': True, 'with_pos': False, 'min_channels': 256, 'drop_rate': 0.0}
            self.merge_query_inter = QueryInteractionX(in_channels=256, mid_channels=256, **query_trans)
        self._prev_param_snapshot = None
        if replace_bn_with_ln:
            replace_bn(self)
        self.debug_mode = debug_mode
        self.dino_enable = bool(dino_enable)
        self.dino_require = bool(dino_require)
        self.dino_debug = bool(dino_debug)
        self.dino_pose_mode = str(dino_pose_mode)
        self.dino_valid_rate_warn_thr = float(dino_valid_rate_warn_thr)
        self._dino_warn_count = 0
        self._last_dino_valid_rate = None
        self._last_dino_valid_rate_per_sample = []
        self._last_dino_point_feats = []
        self._last_dino_point_valids = []
        self.dgi_enable = bool(dgi_enable)
        self.dgi_dino_thr = float(dgi_dino_thr)
        self.dgi_weight = float(dgi_weight)
        self.dgi_cluster_thr = float(dgi_cluster_thr)
        self.dgi_debug = bool(dgi_debug)
        self._dgi_debug_count = 0
        self._dgi_merge_debug_count = 0
        self._dgi_pred_debug_count = 0
        self._last_dgi_stats = []
        self.init_weights()
    

        sp_feats_sem = None
        if x_sem is not None:
            x_sem = x_sem.slice(field)
            sem_feat = scatter_mean(x_sem.features, sp_idx, dim=0)
            sp_feats_sem = []
            for i in range(len(n_super_points)):
                begin = sum(n_super_points[:i])
                end = sum(n_super_points[:i + 1])
                sp_feats_sem.append(sem_feat[begin: end])
        elif getattr(self.decoder, '_dual_path', False):
            sp_feats_sem = []
            for i in range(len(n_super_points)):
                begin = sum(n_super_points[:i])
                end = sum(n_super_points[:i + 1])
                sp_feats_sem.append(x[begin: end, :-3])

        if self.use_temporal_loss and self.inst_dict is not None:
            self.inst_dict['best_obj_ids_list'] = best_obj_ids_list
        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list, sp_feats_sem # [N_segment, 96], [20000, 99], [20000, 1], [N_segment, 3]

    def _pool_last_dino_to_superpoints(self, current_sp_pts_mask):
        """Pool cached point-level DINO features to current superpoints.

        Returns a list aligned with batch samples. Invalid projected points are
        excluded from the average. This is used by DGI to reduce VFM-induced
        over-segmentation before query initialization.
        """
        if not self.dgi_enable:
            return None
        feats_list = getattr(self, '_last_dino_point_feats', [])
        valid_list = getattr(self, '_last_dino_point_valids', [])
        if not feats_list or not valid_list:
            return None
        pooled = []
        for batch_idx, sp_mask in enumerate(current_sp_pts_mask):
            if batch_idx >= len(feats_list) or feats_list[batch_idx] is None:
                pooled.append(None)
                continue
            feats = feats_list[batch_idx]
            valid = valid_list[batch_idx]
            device = sp_mask.device if torch.is_tensor(sp_mask) else feats.device
            sp = sp_mask.to(device=device, dtype=torch.long) if torch.is_tensor(sp_mask) else torch.as_tensor(sp_mask, device=device, dtype=torch.long)
            feats = feats.to(device=device, dtype=torch.float32)
            valid = valid.to(device=device, dtype=torch.float32)
            n_sp = int(sp.max().item()) + 1 if sp.numel() else 0
            if n_sp <= 0:
                pooled.append(None)
                continue
            denom = scatter(valid, sp, dim=0, dim_size=n_sp, reduce='sum').clamp(min=1.0)
            numer = scatter(feats * valid[:, None], sp, dim=0, dim_size=n_sp, reduce='sum')
            sp_dino = numer / denom[:, None]
            sp_dino = F.normalize(sp_dino, p=2, dim=-1, eps=1e-6)
            pooled.append(sp_dino)
        return pooled

    def merge_superpixels_extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i, overlap_threshold=0.99):
        """Merge superpixels and extract features once.

        This integrates feature extraction with superpixel merging to avoid
        recomputing backbone/neck. It first pools by current SPs to decide
        merges, updates masks, then pools again to produce final features.

        Returns: features, point_features, all_xyz_w, sp_xyz_list
        """
        # 1) image/DINO features (same as extract_feat) — support no-image cases
        dino_feats = None
        if self.dino_enable or ('dino_feat_maps' in batch_inputs_dict):
            dino_feats = self._build_dino_fpn_cached(batch_inputs_dict, batch_data_samples, frame_i)
            if dino_feats is None and self.dino_require:
                raise RuntimeError(f'[DINO][strict] failed to build cached DINO FPN in merge path at frame={frame_i}')
        use_images = ('img_paths' in batch_inputs_dict) and hasattr(self, 'img_backbone')
        if use_images:
            with torch.no_grad():
                img_features = []
                for img_paths in batch_inputs_dict['img_paths']:
                    img_features.append(self.img_backbone(img_paths[frame_i])[0])
            img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
            for img_meta in img_metas:
                img_meta['depth2img'] = img_meta['depth2img'][frame_i]

        # 2) construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates

        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # 3) backbone + neck (once)
        if use_images:
            x = self.backbone(field.sparse(),
                              partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=img_metas[0]['img_shape']),
                              memory=self.memory if hasattr(self,'memory') else None)
        else:
            x = self.backbone(field.sparse(), dino_feats=dino_feats,
                              memory=self.memory if hasattr(self,'memory') else None)
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c, f], dim=-1) for c, f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        # 4) pool with current superpixels to compute merging decisions
        sp_pts_masks, n_super_points = [], []
        current_sp_pts_mask = [bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples]
        for sp_pts_mask in current_sp_pts_mask:
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        x_pooled, all_xyz_w_orig = self.pool(x, sp_idx, all_xyz, with_xyz=True)

        features_orig, sp_xyz_list_orig = [], []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features_orig.append(x_pooled[begin: end, :-3])
            sp_xyz_list_orig.append(x_pooled[begin: end, -3:])

        sp_dino_list = self._pool_last_dino_to_superpoints(current_sp_pts_mask)

        # predictor for per-SP detections (lightweight path)
        x_detach = [features_orig[i] for i in range(len(features_orig))]
        pred_bboxes, pred_cls_list, new_queries = [], [], []
        queries = self.decoder._get_queries(x_detach, len(current_sp_pts_mask))
        for i in range(len(queries)):
            norm_query = self.decoder.out_norm(queries[i])
            reg_final = self.decoder.out_reg(norm_query)
            reg_cls = self.decoder.out_cls(norm_query)
            reg_cls = reg_cls.softmax(1)
            reg_distance = torch.exp(reg_final[:, 3:6])
            pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)
            pred_cls_list.append(reg_cls)
            pred_bboxes.append(pred_bbox)
            new_queries.append(norm_query)
        x_detach = [new_queries[i] for i in range(len(new_queries))]

        valid_sps_list = []
        for batch_idx in range(len(pred_bboxes)):
            labels = pred_cls_list[batch_idx].argmax(dim=1)
            bg = pred_cls_list[batch_idx].shape[1] - 1
            labels_mask = ((labels[:, None] == labels) & (labels[:, None] != bg) & (labels != bg)[:, None])
            det_bboxes = pred_bboxes[batch_idx].clone()
            det_bboxes[:, :3] += sp_xyz_list_orig[batch_idx][:, :3]
            pos_embedding = self.merge_box_trans(det_bboxes)
            obj_embedding1, obj_embedding2 = self.merge_query_inter(x_detach[batch_idx], x_detach[batch_idx], pos_embedding)
            merge_rel_dist = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes))
            merge_rel_dist = merge_rel_dist.unsqueeze(-1)
            merge_geometry_embedding = self.merge_dist_embed(merge_rel_dist)
            merge_appear_embedding = obj_embedding1[:, None] * obj_embedding2[None]
            merge_fused_embedding = self.merge_fusion(merge_appear_embedding, merge_geometry_embedding)
            merge_det_mat = self.merge_embed_trans(merge_fused_embedding).sum(-1)
            m = merge_det_mat.sigmoid()
            iou_map = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes), mode='iou')
            merge_score = iou_map * m * labels_mask.float()
            dgi_pairs = 0
            if self.dgi_enable and sp_dino_list is not None and batch_idx < len(sp_dino_list) and sp_dino_list[batch_idx] is not None:
                sp_dino = sp_dino_list[batch_idx].to(device=det_bboxes.device, dtype=torch.float32)
                if sp_dino.shape[0] == merge_score.shape[0]:
                    dino_aff = (sp_dino @ sp_dino.t()).clamp(min=0.0, max=1.0)
                    dino_aff = torch.where(dino_aff >= self.dgi_dino_thr, dino_aff, torch.zeros_like(dino_aff))
                    dgi_pairs = int((dino_aff > 0).sum().item() - dino_aff.shape[0])
                    # For inference-only DGI we must not rely on the learnable
                    # AutoSeg3D merge head, because a stage1 checkpoint may not have
                    # trained those parameters. dgi_weight=1.0 gives pure DINO
                    # affinity; lower values blend in the learned merge score.
                    dgi_boost = self.dgi_weight * dino_aff + (1.0 - self.dgi_weight) * m
                    merge_score = iou_map * dgi_boost * labels_mask.float()
            cluster_both = cluster_complete_link(merge_score, self.dgi_cluster_thr if self.dgi_enable else 0.5)
            if self.dgi_enable:
                self._last_dgi_stats.append(dict(
                    frame=int(frame_i), batch=int(batch_idx), n_sp=int(merge_score.shape[0]),
                    dino_pairs=max(dgi_pairs, 0), clusters=sum(1 for c in cluster_both if len(c) > 1)))
                if self.dgi_debug and self._dgi_debug_count < 40:
                    print(f"[DGI] frame={frame_i} batch={batch_idx} n_sp={merge_score.shape[0]} dino_pairs={max(dgi_pairs, 0)} clusters={sum(1 for c in cluster_both if len(c) > 1)} thr={self.dgi_dino_thr:.2f}")
                    self._dgi_debug_count += 1
            valid_sps_list.append(cluster_both)

        # generate merged masks and update in-place
        for batch_idx in range(len(current_sp_pts_mask)):
            sp_pts_mask = current_sp_pts_mask[batch_idx]
            merged_mask = sp_pts_mask.clone()
            merged_groups = []
            for valid_sps in valid_sps_list[batch_idx]:
                if len(valid_sps) > 1:
                    merged_groups.append(valid_sps)
                    for sp_id in valid_sps:
                        merged_mask[sp_pts_mask == sp_id] = valid_sps[0]
            all_original_ids = sp_pts_mask.unique().tolist()
            merged_ids = set()
            for group in merged_groups:
                merged_ids.update(group)
            unmerged_ids = [id for id in all_original_ids if id not in merged_ids]
            new_id = 0
            id_mapping = {}
            for group in merged_groups:
                for old_id in group:
                    id_mapping[old_id] = new_id
                new_id += 1
            for old_id in unmerged_ids:
                id_mapping[old_id] = new_id
                new_id += 1
            final_mask = merged_mask.clone()
            for old_id, nid in id_mapping.items():
                final_mask[merged_mask == old_id] = nid
            if self.dgi_enable and self.dgi_debug and self._dgi_merge_debug_count < 40:
                before_n = int(sp_pts_mask.max().item() + 1) if sp_pts_mask.numel() > 0 else 0
                after_n = int(final_mask.max().item() + 1) if final_mask.numel() > 0 else 0
                merged_sp_count = sum(len(g) for g in merged_groups)
                max_group = max([len(g) for g in merged_groups], default=1)
                print(f"[DGI_MERGE] frame={frame_i} batch={batch_idx} before_sp={before_n} after_sp={after_n} merged_groups={len(merged_groups)} merged_sps={merged_sp_count} max_group={max_group}")
                self._dgi_merge_debug_count += 1
            batch_data_samples[batch_idx].gt_pts_seg.sp_pts_mask[frame_i] = final_mask

        # 5) pool with merged superpixels to produce final features
        sp_pts_masks_new, n_super_points_new = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i]
            sp_pts_masks_new.append(sp_pts_mask + sum(n_super_points_new))
            n_super_points_new.append(sp_pts_mask.max() + 1)
        sp_idx_new = torch.cat(sp_pts_masks_new)
            query_x = self.muti_scale_query(sp_xyz, x, self.query_memory, self.query_memory, self.pos_memory)
            x = [x[i] + query_x[i] for i in range(len(x))]
            x = [self.query_memory_relu(x[i]) for i in range(len(x))]
        with torch.no_grad():
            detach_query = [x[i].clone().detach() for i in range(len(x))]
            detach_pos = [sp_xyz[i].clone().detach() for i in range(len(sp_xyz))]
            self.query_memory = detach_query
            self.pos_memory = detach_pos
        return x
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        assert len(batch_data_samples) == 1
        results, query_feats_list, sem_preds_list, sp_xyz_list, bboxes_list, cls_preds_list = [], [], [], [], [], []
        num_frames = batch_inputs_dict['points'][0].shape[0]
        if hasattr(self, 'memory'):
            self.memory.reset()
        if self.use_query_memory:
            self.reset_query_memory()
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_mask_memory = None
            self.before_sp_xyz = None
        if self.use_temporal_loss:
            self.inst_dict = None
        if self.use_mot:
            self.current_max_track_id = 0
        for frame_i in range(num_frames):
            ## Backbone + SP merge (optional)  -> features, point_features, all_xyz_w, sp_xyz
            if self.merge_sp_masks:
                # Integrated path: merge SPs and extract features once. DINO dual side branch is disabled on this legacy path until stage2-specific integration.
                x, point_features, all_xyz_w, sp_xyz = self.merge_superpixels_extract_feat(
                    batch_inputs_dict, batch_data_samples, frame_i)
                sp_feats_sem = None
            else:
                x, point_features, all_xyz_w, sp_xyz, sp_feats_sem = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            ## Query
            if self.use_query_memory:
                x = self.query_memory_aggregation_predict(x, sp_xyz)
            if self.use_self_attn:
                query_self = self.muti_scale_self_attn(sp_xyz, x, x, x, sp_xyz)
                x = [x[i] + query_self[i] for i in range(len(x))]
                x = [self.self_attn_relu(x[i]) for i in range(len(x))]
            if self.use_mot:
                device = x[0].device
                is_last = frame_i == num_frames - 1 

                if frame_i == 0:
                    track_instances = self._init_query_test(x, mot_type=self.mot_type, mode='test')
                    active_queries_num =  [q.shape[0] for q in track_instances.queries]
                else:
                    init_track_instances: Instances = track_instances
