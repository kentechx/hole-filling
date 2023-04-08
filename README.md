# HoleFillingPy

A dirty implementation of mesh hole filling with [libigl](https://github.com/libigl/libigl-python-bindings).

<img src="./docs/hole_filling.png" width="600">

## Installation

```bash
pip install hole-filling
```

## Usage

Simply fill all holes.

```python
import numpy as np
import igl
from hole_filling import triangulate_refine_fair

vs, fs, _ = igl.read_off("tests/examples/bunny_holes.off")
out_vs, out_fs = triangulate_refine_fair(vs, fs)

colors = np.ones((len(out_vs), 3))
colors[np.arange(len(vs), len(out_vs))] = [0, 0, 1]  # added vertices are blue

igl.write_off("bunny_hole_filling.off", out_vs, out_fs, colors)
```

You can fill holes with fine controls, e.x. keeping the longest boundary.

```python
import numpy as np
import igl
from hole_filling import close_holes, triangulate_refine_fair

vs, fs, _ = igl.read_off("tests/examples/face_holes.off")
loop = igl.boundary_loop(fs)
length = np.linalg.norm(vs[loop[:-1]] - vs[loop[1:]], axis=1).sum()
# tweak the parameters to see the difference
out_vs, out_fs = triangulate_refine_fair(vs, fs, hole_len_thr=length - 0.1, density_factor=2, fair_alpha=0.5)
igl.write_off("face_hole_filling.off", out_vs, out_fs, np.ones_like(out_vs) * 0.7)
```

## Reference

1. Liepa, Peter. "Filling holes in meshes." Proceedings of the 2003 Eurographics/ACM SIGGRAPH symposium on Geometry
   processing. 2003.
2. Jacobson, Alec, and Daniele Panozzo. "Libigl: Prototyping geometry processing research in c++." SIGGRAPH Asia 2017
   courses. 2017. 1-172.
