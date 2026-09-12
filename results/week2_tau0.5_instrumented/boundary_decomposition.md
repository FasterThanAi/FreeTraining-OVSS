# 9.1a — Boundary vs interior decomposition of the residual

- tiles: **1669**  |  τ: **0.5**
- real-class pixels: **1,089,045,589**
- assigned to background: **323,084,415** (29.67%)

`band` = within k pixels of a GT class boundary (4-conn seam, dilated).
`enrichment` = (share of discard in band) / (share of area in band).
Enrichment ≈ 1 means discard is spread uniformly and the band explains nothing.

| band width k | band share of area | band share of discard | enrichment | interior discard (addressable) |
|---|---|---|---|---|
| 1 | 1.93% | 2.70% | **1.40×** | 314,351,222 (28.86% of real-class) |
| 2 | 3.83% | 5.23% | **1.37×** | 306,192,861 (28.12% of real-class) |
| 3 | 5.71% | 7.59% | **1.33×** | 298,554,543 (27.41% of real-class) |
| 5 | 9.37% | 11.87% | **1.27×** | 284,726,499 (26.14% of real-class) |
| 10 | 17.62% | 20.35% | **1.16×** | 257,336,476 (23.63% of real-class) |

## Per class, at k=10

Which classes lose their pixels on seams, and which lose whole regions.

| Class | total discard | in band | in interior | band share of its discard | enrichment |
|---|---|---|---|---|---|
| building | 22,981,980 | 10,692,829 | 12,289,151 | 46.5% | **1.34×** |
| road | 18,407,996 | 10,368,884 | 8,039,112 | 56.3% | **1.45×** |
| water | 64,295,090 | 9,946,878 | 54,348,212 | 15.5% | **1.15×** |
| barren | 18,569,281 | 5,197,993 | 13,371,288 | 28.0% | **1.39×** |
| forest | 43,445,156 | 11,684,132 | 31,761,024 | 26.9% | **0.99×** |
| agricultural | 155,384,912 | 17,857,223 | 137,527,689 | 11.5% | **1.33×** |

A class with high enrichment loses its pixels on seams (annotation-boundary effects, thin structures). A class near 1.00× loses whole regions — that is the population a region-level prior can actually recover.

## Verdict

At k=10, enrichment is **1.16×** — mild. Boundaries carry somewhat more discard than their area share, but most of the residual is interior. Report both; the headline survives with a sentence of qualification.

> Note the band grows fast: at k=10 a large share of a tile is within 10px of some boundary, so a high band share at large k is partly definitional. Read the enrichment column, which normalises for exactly that, not the raw share.
