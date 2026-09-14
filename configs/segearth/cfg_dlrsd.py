_base_ = './base_config.py'

# DLRSD -- the dense-labelling extension of UC Merced. 2100 images, 256x256,
# 17 classes, every pixel labelled.
#
# ⛔ SegEarth-OV3 publishes NO DLRSD row and ships no config for it, so the
# vocabulary, the threshold and the discard sink below are OURS. All three are
# fixed in prereg/predict_dlrsd.md, committed before any inference, precisely
# because each could otherwise be tuned after seeing the result.
#
# ⚠️ THERE IS NO REPRODUCTION GATE HERE. Every other dataset in this project is
# checked against a released number (LoveDA -0.02, Potsdam +0.03, UAVid +2.16
# and reported as unexplained). DLRSD has none, so a preparation error cannot
# be caught the usual way. The substitutes are the pixel-exact label accounting
# in prepare_dlrsd.py and the category fingerprint in dlrsd_class_map.py, and
# both are weaker than a gate.

model = dict(
    classname_path='./configs/cls_dlrsd.txt',

    # Reference operating point. 0.1 is the lowest published tau in the project
    # -- Potsdam and OpenEarthMap both use it -- chosen for cross-dataset
    # comparability, NOT because it is good for DLRSD. ⭐ The baseline the
    # method must actually beat is the best GLOBAL tau fitted on the same
    # calibration tiles (prereg 1d), which is computed from the cache.
    prob_thd=0.1,

    # ⭐⭐ THE UNSCORED SINK. The class list has 17 entries, indexed 0..16, so
    # index 17 has no prompt, no ground-truth pixel and no column in the
    # metric. mmseg's intersect_and_union histograms predictions over
    # [0, num_classes-1] = [0, 16], so a prediction of 17 is counted nowhere:
    # it contributes to no class's predicted area and to no intersection,
    # while the true class still counts it in its own area.
    #
    #   => a discarded pixel becomes a FALSE NEGATIVE for its true class and a
    #      false positive for nothing.
    #
    # Everywhere else in this project the discard target is a real, scored
    # catch-all, so thresholding a wrong pixel costs that class's precision.
    # DLRSD has no catch-all to absorb it (0.00% of ground truth), and
    # inventing a `background` prompt would make every win by that prompt a
    # pure loss, since no GT pixel can ever be background.
    #
    # ⚠️ Verified by scripts/test_dlrsd_sink.py against mmseg's own metric.
    # Do not change this number without re-running that test.
    bg_idx=17,

    # ⛔ confidence_threshold is DELIBERATELY NOT SET, so it inherits from
    # _base_. It is SAM 3's DECODER threshold and changes seg_logits
    # themselves -- hardcoding 0.5 in a generated config once meant a dataset
    # was silently evaluated with a different model from the baseline
    # (@ARGMAX_SCALING_RESULTS.md). Record whatever _base_ supplies; do not
    # guess a value here.
)

dataset_type = 'DLRSDDataset'
data_root = 'data/DLRSD'

# Labels ship as mode-P (palette) PNGs carrying indices 1..17 with no 0.
# prepare_dlrsd.py rewrites them as mode-L so no image backend can expand the
# palette to RGB behind our backs; reduce_zero_label then maps 1..17 -> 0..16.
# No pixel is 0, so nothing becomes ignore -- unlike LoveDA, where 0 is real
# no-data covering ~2.6% of pixels.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
