# Appended to SegEarth-OV-3/custom_datasets.py by scripts/install_dlrsd.sh.
# Kept here so the definition is in version control rather than living only
# inside the baseline clone.
#
# ⭐ The palette is NOT copied from a paper. It is read off DLRSD's own label
# PNGs (scripts/inspect_dataset.py --palette) and the class ladder it implies
# was then CONFIRMED against the data by scripts/dlrsd_class_map.py: `airplane`
# is 100.0% inside airplane*, `dock` 100.0% inside harbor*, `tanks` 100.0%
# inside storagetanks*, and each appears in exactly its category's 100 images.
# WEEK3 §11 records why this matters -- a hardcoded ladder would have labelled
# `grass` as `building` on OpenEarthMap and printed a clean table with the
# wrong row names, crashing nothing.
@DATASETS.register_module()
class DLRSDDataset(BaseSegDataset):
    """DLRSD: the dense-labelling extension of UC Merced.

    2100 images, 256x256, 17 classes, 21 scene categories of 100 images each.
    Mask values run 1..17 with 0 absent from all 2100 files, so
    ``reduce_zero_label=True`` maps them to 0..16 and nothing becomes ignore.

    ⭐ There is NO catch-all class. Every pixel carries a real class, so full
    mIoU equals catch-all-excluded mIoU by construction.
    """

    METAINFO = dict(
        classes=('airplane', 'bare soil', 'buildings', 'cars', 'chaparral',
                 'court', 'dock', 'field', 'grass', 'mobile home', 'pavement',
                 'sand', 'sea', 'ship', 'tanks', 'trees', 'water'),
        palette=[[166, 202, 240], [128, 128, 0], [0, 0, 128], [255, 0, 0],
                 [0, 128, 0], [128, 0, 0], [255, 233, 233], [160, 160, 164],
                 [0, 128, 128], [90, 87, 255], [255, 255, 0], [255, 192, 0],
                 [0, 0, 255], [255, 0, 192], [128, 0, 128], [0, 255, 0],
                 [0, 255, 255]])

    def __init__(self, img_suffix='.png', seg_map_suffix='.png',
                 reduce_zero_label=True, **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=reduce_zero_label, **kwargs)
