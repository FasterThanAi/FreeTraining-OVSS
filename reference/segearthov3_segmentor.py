# =============================================================================
# INSTRUMENTED COPY of SegEarth-OV-3's segearthov3_segmentor.py
#
# Upstream: https://github.com/earth-insights/SegEarth-OV-3
# Vendored here to pin the exact code that produced our 47.38 mIoU baseline.
#
# Local changes are marked  # <<< INSTRUMENTATION  and are OBSERVE-ONLY:
# they record `presence_score` (S_pres) per query per view, and change no
# tensor that feeds a prediction. Search that marker to see every edit.
#
# Why: P_final = P_fused * S_pres (line ~95). S_pres is a hard per-class,
# per-view ceiling, but it is a local variable consumed and discarded, so the
# eval path cannot observe it. WEEK1_RESULTS.md 9.2 needs it across all 1669
# tiles to decide whether presence collapse is systematic or a single anecdote.
#
# VALIDATION GATE: an instrumented run must still produce mIoU 47.37 and
# 29.68% discard at tau=0.5. If either moves, these edits changed behaviour.
# =============================================================================
import numpy as np                                    # <<< INSTRUMENTATION
import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@MODELS.register_module()
class SegEarthOV3Segmentation(BaseSegmentor):
    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 class_scale=None,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,
                 **kwargs):
        super().__init__()
        
        self.device = device
        # Initialize SAM3 model
        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='weights/sam3/sam3.pt', 
            device="cuda"
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.bg_idx = bg_idx
        self.prob_thd = None
        self.prob_thd_vec = None
        self.set_prob_thd(prob_thd)
        self.class_scale = None
        self.set_class_scale(class_scale)
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder

        # <<< INSTRUMENTATION: presence-score capture.
        # presence_log accumulates one row per *view* (a whole image, or one
        # sliding-window crop). predict() resets it per image and stacks it
        # into last_presence with shape (n_views, num_queries).
        # NOTE queries != classes: cls_loveda.txt has 7 classes but 11 queries
        # (building,house / barren,bareland,soil / forest,tree). Collapse to
        # classes with max-over-synonyms, mirroring the .max(1) on line ~183.
        self.presence_log = []
        self.last_presence = np.zeros((0, self.num_queries), dtype=np.float32)
        self.last_fused = None       # <<< INSTRUMENTATION: P_fused, pre-gating
        self.last_inst = None        # <<< CROSS-HEAD: P_inst_agg alone
        self.last_sem = None         # <<< CROSS-HEAD: P_sem alone

    # <<< PER-CLASS TAU: `prob_thd` may be a scalar (the published baseline,
    # unchanged) or a sequence of length num_cls giving one threshold per class.
    # The rule is the same either way -- a pixel is assigned to bg_idx when the
    # winning score falls below the threshold -- but which threshold applies is
    # chosen by the ARGMAX CLASS, not fixed globally. That is the only change.
    #
    # Scalar is the default and reproduces 47.37 exactly, so the validation gate
    # is preserved by construction rather than by re-checking.
    def set_prob_thd(self, prob_thd):
        if isinstance(prob_thd, (list, tuple, np.ndarray, torch.Tensor)):
            vec = torch.as_tensor(prob_thd, dtype=torch.float32).flatten()
            if vec.numel() != self.num_cls:
                raise ValueError(
                    f'prob_thd has {vec.numel()} entries but the model has '
                    f'{self.num_cls} classes. A silently misaligned threshold '
                    f'vector would apply water\'s tau to forest and still '
                    f'produce a plausible mIoU, so this is fatal.')
            if not torch.all((vec >= 0) & (vec <= 1)):
                raise ValueError(f'prob_thd out of [0, 1]: {vec.tolist()}')
            self.prob_thd_vec = vec.to(self.device)
            self.prob_thd = None
            print('  per-class prob_thd: '
                  + ', '.join(f'{t:.3f}' for t in vec.tolist()))
        else:
            self.prob_thd_vec = None
            self.prob_thd = float(prob_thd)

    # <<< ARGMAX SCALING: one multiplier per class, applied to the scores BEFORE
    # the argmax and nowhere else.
    #
    # Why this is a different lever from prob_thd, and not a reparameterisation
    # of it: a per-class scale applied AFTER the argmax is monotone, so it folds
    # into the threshold and buys nothing. Applied BEFORE, it changes which class
    # wins -- which no threshold vector can do, because lowering a threshold
    # cannot change an argmax. That is the family per-class tau provably cannot
    # reach.
    #
    # `None` is the default and leaves predict() on the ORIGINAL code path, so
    # the 47.38 gate is preserved by construction rather than by re-checking.
    def set_class_scale(self, class_scale):
        if class_scale is None:
            self.class_scale = None
            return
        vec = torch.as_tensor(class_scale, dtype=torch.float32).flatten()
        if vec.numel() != self.num_cls:
            raise ValueError(
                f'class_scale has {vec.numel()} entries but the model has '
                f'{self.num_cls} classes. A misaligned scale vector would apply '
                f'water\'s multiplier to forest and still produce a plausible '
                f'mIoU, so this is fatal.')
        if not torch.all(vec > 0):
            raise ValueError(
                f'class_scale must be strictly positive: {vec.tolist()}. A zero '
                f'or negative entry does not down-weight a class, it deletes or '
                f'inverts it.')
        self.class_scale = vec.to(self.device)
        print('  class_scale: ' + ', '.join(f'{s:.3f}' for s in vec.tolist()))


    def _inference_single_view(self, image):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)
        # <<< INSTRUMENTATION: P_fused, i.e. max(P_sem, P_inst_agg) BEFORE the
        # presence multiply. conf in the cache is P_final = P_fused * S_pres, so
        # the two factors are entangled and cannot be separated after the fact.
        # WEEK1_RESULTS 9.2 tile 3487 is the reason this matters: P_sem was
        # effectively certain road was present (+10.13 logit) while S_pres = 0.076
        # crushed P_final below any tau. If P_fused separates recoverable pixels
        # from true background where P_final cannot (AUC 0.582), tau_low exists
        # after all -- just not in the gated score.
        fused_logits = torch.zeros((self.num_queries, h, w), device=self.device)
        # <<< CROSS-HEAD: P_inst_agg and P_sem SEPARATELY. `fused = max(sem, inst)`
        # destroys the distinction, so nothing downstream can ask whether the two
        # heads agreed -- and cross-head agreement is the leading label-free
        # candidate for per-class precision (WEEK3 9a: the oracle threshold is set
        # by precision, which is label-derived). -inf, not 0, is the identity for
        # max, so a class the head never fires on stays distinguishable from one
        # it scores at exactly 0.
        inst_logits = torch.full((self.num_queries, h, w), float('-inf'), device=self.device)
        sem_logits_all = torch.full((self.num_queries, h, w), float('-inf'), device=self.device)
        # <<< INSTRUMENTATION: NaN = "not recorded" (distinct from a real 0.0)
        view_presence = np.full(self.num_queries, np.nan, dtype=np.float32)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                if self.use_transformer_decoder:
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            instance_score = inference_state['object_score'][inst_id]
                            # instance_mask = inference_state['masks'][inst_id].squeeze()
                            
                            # Handle potential dimension mismatch if SAM3 output differs slightly
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape), 
                                    size=(h, w), 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze()

                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits * instance_score)
                            inst_logits[query_idx] = torch.max(              # <<< CROSS-HEAD
                                inst_logits[query_idx], instance_logits * instance_score)
                    
                if self.use_sem_seg:
                    semantic_logits = inference_state['semantic_mask_logits']
                    if semantic_logits.shape != (h, w):
                            semantic_logits = F.interpolate(
                                semantic_logits, 
                                size=(h, w), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze()
                    
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)
                    sem_logits_all[query_idx] = semantic_logits           # <<< CROSS-HEAD
                
                # <<< INSTRUMENTATION: snapshot AFTER dual-head fusion, BEFORE
                # gating. Unconditional, so --no-presence still records it.
                fused_logits[query_idx] = seg_logits[query_idx]

                if self.use_presence_score:
                    presence = inference_state["presence_score"]        # <<< INSTRUMENTATION
                    try:                                                # <<< INSTRUMENTATION
                        view_presence[query_idx] = float(presence)      # <<< INSTRUMENTATION
                    except Exception:                                   # <<< INSTRUMENTATION
                        pass       # never let bookkeeping break a 25-min run
                    seg_logits[query_idx] = seg_logits[query_idx] * presence

        self.presence_log.append(view_presence)                         # <<< INSTRUMENTATION
        # -inf would poison the sliding-window sum; a head that never fired is
        # reported at the floor instead, which is what "no evidence" means here.
        inst_logits = torch.nan_to_num(inst_logits, neginf=0.0)          # <<< CROSS-HEAD
        sem_logits_all = torch.nan_to_num(sem_logits_all, neginf=0.0)    # <<< CROSS-HEAD
        return seg_logits, fused_logits, inst_logits, sem_logits_all     # <<< INSTRUMENTATION

    def slide_inference(self, image, stride, crop_size):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        
        # Initialize accumulators
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        fused = torch.zeros((self.num_queries, h_img, w_img), device=self.device)  # <<< INSTRUMENTATION
        inst = torch.zeros((self.num_queries, h_img, w_img), device=self.device)   # <<< CROSS-HEAD
        sem = torch.zeros((self.num_queries, h_img, w_img), device=self.device)    # <<< CROSS-HEAD
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)
        
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                
                # Adjust start points to ensure crop size is valid at boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                # Crop via PIL
                crop_img = image.crop((x1, y1, x2, y2))
                
                # Inference on crop
                crop_seg_logit, crop_fused, crop_inst, crop_sem = \
                    self._inference_single_view(crop_img)
                
                # Accumulate results
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                fused[:, y1:y2, x1:x2] += crop_fused                    # <<< INSTRUMENTATION
                inst[:, y1:y2, x1:x2] += crop_inst                      # <<< CROSS-HEAD
                sem[:, y1:y2, x1:x2] += crop_sem                        # <<< CROSS-HEAD
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        
        preds = preds / count_mat
        fused = fused / count_mat                                       # <<< INSTRUMENTATION
        inst = inst / count_mat                                         # <<< CROSS-HEAD
        sem = sem / count_mat                                           # <<< CROSS-HEAD
        return preds, fused, inst, sem                                  # <<< INSTRUMENTATION

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            # Fallback for meta info construction
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        for i, meta in enumerate(batch_img_metas):
            # Load original image to preserve details for SAM3
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            self.presence_log = []       # <<< INSTRUMENTATION: reset per image

            # Determine inference mode
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                seg_logits, fused_logits, inst_logits, sem_logits = self.slide_inference(
                    image, self.slide_stride, self.slide_crop)
            else:
                seg_logits, fused_logits, inst_logits, sem_logits = \
                    self._inference_single_view(image)
            self.last_fused = fused_logits                              # <<< INSTRUMENTATION
            self.last_inst = inst_logits                                # <<< CROSS-HEAD
            self.last_sem = sem_logits                                  # <<< CROSS-HEAD

            # <<< INSTRUMENTATION: (n_views, num_queries). One row if this image
            # was a single forward pass, one row per crop under sliding window.
            self.last_presence = (np.stack(self.presence_log) if self.presence_log
                                  else np.zeros((0, self.num_queries), dtype=np.float32))

            # Resize to original shape if necessary (e.g. padding effects)
            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0), 
                    size=ori_shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Post-processing
            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            # <<< ARGMAX SCALING: the argmax reads the SCALED scores; the
            # threshold below reads the RAW ones. Keeping `max_vals` raw is what
            # confines the scale to the reordering -- scaling the score the
            # threshold sees would just reparameterise prob_thd.
            if self.class_scale is None:
                seg_pred = torch.argmax(seg_logits, dim=0)
                max_vals = seg_logits.max(0)[0]          # original line, untouched
            else:
                seg_pred = torch.argmax(
                    seg_logits * self.class_scale.view(-1, 1, 1), dim=0)
                max_vals = seg_logits.gather(0, seg_pred.unsqueeze(0)).squeeze(0)
            
            # Apply probability threshold
            # <<< PER-CLASS TAU: index the threshold by the predicted class.
            # With a scalar this is bit-identical to the published line.
            thd = (self.prob_thd if self.prob_thd_vec is None
                   else self.prob_thd_vec[seg_pred])
            seg_pred[max_vals < thd] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })
            
        return data_samples
    
    def _forward(data_samples):
            """
        """
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices