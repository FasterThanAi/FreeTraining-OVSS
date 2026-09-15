# DLRSD with three corrected prompts. See prereg/predict_dlrsd_vocabulary.md.
#
# ⛔ This is a SEPARATE ARM, not a replacement. The method's DLRSD result
# (37.27 -> 44.42) is measured on the PUBLISHED vocabulary and stays that way;
# anything measured here is reported as its own result, exactly as UAVid's
# one-word +3.53 was.
#
# Three lines differ from cls_dlrsd.txt and nothing else does:
#     chaparral   -> shrubs       (regional biome term -> the common word)
#     mobile home -> trailer      (compound -> the ordinary term)
#     field       -> crop field   (ambiguous -> disambiguated by one modifier)
#
# ⭐ Arity is unchanged -- one prompt per line, no synonyms added -- so this
# cannot be confused with prompt ensembling (WEEK3 §7b).
#
# ⭐ Each class is an independent forward pass with its own text prompt, so the
# other FOURTEEN channels should come back bit-identical. That is prediction W6
# and it is the control: if untouched classes move, the run is not a controlled
# comparison and nothing else in it can be read.
_base_ = './cfg_dlrsd.py'

model = dict(
    classname_path='./configs/cls_dlrsd_v2.txt',
)
