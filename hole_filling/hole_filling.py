import numpy as np
import igl
import scipy
import scipy.sparse
import scipy.sparse.linalg
from typing import List, Tuple

_epsilon = 1e-16


def close_hole(vs: np.ndarray, fs: np.ndarray, hole_vids, fast=True) -> np.ndarray:
    """
    :param hole_vids: the vid sequence
    :return:
        out_fs:
    """

    def hash_func(edges):
        # edges: (n, 2)
        edges = np.core.defchararray.chararray.encode(edges.astype('str'))
        edges = np.concatenate([edges[:, 0:1], np.full_like(edges[:, 0:1], "_", dtype=str), edges[:, 1:2]], axis=1)
        edges_hash = np.core.defchararray.add(np.core.defchararray.add(edges[:, 0], edges[:, 1]), edges[:, 2])
        return edges_hash

    # create edge hash
    if not fast:
        edges = igl.edges(fs)
        edges_hash = hash_func(edges)

    hole_vids = np.array(hole_vids)
    if len(hole_vids) < 3:
        return fs.copy()

    if len(hole_vids) == 3:
        # fill one triangle
        out_fs = np.concatenate([fs, hole_vids[::-1][None]], axis=0)
        return out_fs

    # heuristically divide the hole
    queue = [hole_vids[::-1]]
    out_fs = []
    while len(queue) > 0:
        cur_vids = queue.pop(0)
        if len(cur_vids) == 3:
            out_fs.append(cur_vids)
            continue

        # current hole
        hole_edge_len = np.linalg.norm(vs[np.roll(cur_vids, -1)] - vs[cur_vids], axis=1)
        hole_len = np.sum(hole_edge_len)
        min_concave_degree = np.inf
        tar_i, tar_j = -1, -1
        for i in range(len(cur_vids)):
            eu_dists = np.linalg.norm(vs[cur_vids[i]] - vs[cur_vids], axis=1)
            if not fast:
                # check if the edge exists
                _edges = np.sort(np.stack([np.tile(cur_vids[i], len(cur_vids)), cur_vids], axis=1), axis=1)
                _edges_hash = hash_func(_edges)
                eu_dists[np.isin(_edges_hash, edges_hash, assume_unique=True)] = np.inf

            geo_dists = np.roll(np.roll(hole_edge_len, -i).cumsum(), i)
            geo_dists = np.roll(np.minimum(geo_dists, hole_len - geo_dists), 1)
            concave_degree = eu_dists / (geo_dists ** 2 + _epsilon)
            concave_degree[i] = -np.inf  # there may exist two duplicate vertices

            _idx = 1
            j = np.argsort(concave_degree)[_idx]
            while min((j + len(cur_vids) - i) % len(cur_vids), (i + len(cur_vids) - j) % len(cur_vids)) <= 1:
                _idx += 1
                j = np.argsort(concave_degree)[_idx]

            if concave_degree[j] < min_concave_degree:
                min_concave_degree = concave_degree[j]
                tar_i, tar_j = min(i, j), max(i, j)

        queue.append(cur_vids[tar_i:tar_j + 1])
        queue.append(np.concatenate([cur_vids[tar_j:], cur_vids[:tar_i + 1]]))

    out_fs = np.concatenate([fs, np.array(out_fs)], axis=0)
    return out_fs


def close_holes(vs: np.ndarray, fs: np.ndarray, hole_len_thr: float = 10000., fast=True) -> np.ndarray:
    """
    Close holes whose length is less than a given threshold.
    :param edge_len_thr:
    :return:
        out_fs: output faces
    """
    out_fs = fs.copy()
    while True:
        updated = False
        for b in igl.all_boundary_loop(out_fs):
            hole_edge_len = np.linalg.norm(vs[np.roll(b, -1)] - vs[b], axis=1).sum()
            if len(b) >= 3 and hole_edge_len <= hole_len_thr:
                out_fs = close_hole(vs, out_fs, b, fast)
                updated = True

        if not updated:
            break

    return out_fs


def get_mollified_edge_length(vs: np.ndarray, fs: np.ndarray, mollify_factor=1e-5) -> np.ndarray:
    lin = igl.edge_lengths(vs, fs)
    if mollify_factor == 0:
        return lin
    delta = mollify_factor * np.mean(lin)
    eps = np.maximum(0, delta - lin[:, 0] - lin[:, 1] + lin[:, 2])
    eps = np.maximum(eps, delta - lin[:, 0] - lin[:, 2] + lin[:, 1])
    eps = np.maximum(eps, delta - lin[:, 1] - lin[:, 2] + lin[:, 0])
    eps = eps.max()
    lin += eps
    return lin


def mesh_fair_laplacian_energy(vs: np.ndarray, fs: np.ndarray, vids: np.ndarray, alpha=0.05, k=2):
    L, M = robust_laplacian(vs, fs)
    Q = igl.harmonic_weights_integrated_from_laplacian_and_mass(L, M, k)

    a = np.full(len(vs), 0.)  # alpha
    a[vids] = alpha
    a = scipy.sparse.diags(a)
    out_vs = scipy.sparse.linalg.spsolve(a * Q + M - a * M, (M - a * M) @ vs)
    return np.ascontiguousarray(out_vs)


def robust_laplacian(vs, fs, mollify_factor=1e-5) -> Tuple[scipy.sparse.csc_matrix, scipy.sparse.csc_matrix]:
    """
    Get a laplcian with iDT (intrinsic Delaunay triangulation) and intrinsic mollification.
    Ref https://www.cs.cmu.edu/~kmcrane/Projects/NonmanifoldLaplace/NonmanifoldLaplace.pdf
    :param mollify_factor: the mollification factor.
    """
    lin = get_mollified_edge_length(vs, fs, mollify_factor)
    lin, fin = igl.intrinsic_delaunay_triangulation(lin.astype('f8'), fs)
    L = igl.cotmatrix_intrinsic(lin, fin)
    M = igl.massmatrix_intrinsic(lin, fin, igl.MASSMATRIX_TYPE_VORONOI)
    return L, M


def triangulation_refine_leipa(vs: np.ndarray, fs: np.ndarray, fids: np.ndarray, density_factor: float = np.sqrt(2)):
    """
    Refine the triangles using barycentric subdivision and Delaunay triangulation.
    You should remove unreferenced vertices before the refinement.
    See "Filling holes in meshes." [Liepa 2003].

    :return:
        out_vs: (n, 3), the added vertices are appended to the end of the original vertices.
        out_fs: (m, 3), the added faces are appended to the end of the original faces.
        FI: (len(fs), ), face index mapping from the original faces to the refined faces, where
            fs[i] = out_fs[FI[i]], and FI[i]=-1 means the i-th face is deleted.
    """
    out_vs = np.copy(vs)
    out_fs = np.copy(fs)

    if fids is None or len(fids) == 0:
        return out_vs, out_fs, np.arange(len(fs))

    # initialize sigma
    edges = igl.edges(np.delete(out_fs, fids, axis=0))  # calculate the edge length without faces to be refined
    edges = np.concatenate([edges, edges[:, [1, 0]]], axis=0)
    edge_lengths = np.linalg.norm(out_vs[edges[:, 0]] - out_vs[edges[:, 1]], axis=-1)
    edge_length_vids = edges[:, 0]
    v_degrees = np.bincount(edge_length_vids, minlength=len(out_vs))
    v_sigma = np.zeros(len(out_vs))
    v_sigma[v_degrees > 0] = np.bincount(edge_length_vids, weights=edge_lengths,
                                         minlength=len(out_vs))[v_degrees > 0] / v_degrees[v_degrees > 0]
    if np.any(v_sigma == 0):
        v_sigma[v_sigma == 0] = np.median(v_sigma[v_sigma != 0])
        # print("Warning: some vertices have no adjacent faces, the refinement may be incorrect.")

    all_sel_fids = np.copy(fids)
    for _ in range(100):
        # calculate sigma of face centers
        vc_sigma = v_sigma[out_fs].mean(axis=1)  # nf

        # check edge length
        s = density_factor * np.linalg.norm(
            out_vs[out_fs[all_sel_fids]].mean(1, keepdims=True) - out_vs[out_fs[all_sel_fids]], axis=-1)
        cond = np.all(np.logical_and(s > vc_sigma[all_sel_fids, None], s > v_sigma[out_fs[all_sel_fids]]), axis=1)
        sel_fids = all_sel_fids[cond]  # need to subdivide

        if len(sel_fids) == 0:
            break

        # subdivide
        out_vs, added_fs = igl.false_barycentric_subdivision(out_vs, out_fs[sel_fids])

        # update v_sigma after subdivision
        v_sigma = np.concatenate([v_sigma, vc_sigma[sel_fids]], axis=0)
        assert len(v_sigma) == len(out_vs)

        # delete old faces from out_fs and all_sel_fids
        out_fs[sel_fids] = -1
        all_sel_fids = np.setdiff1d(all_sel_fids, sel_fids)

        # add new vertices, faces & update selection
        out_fs = np.concatenate([out_fs, added_fs], axis=0)
        sel_fids = np.arange(len(out_fs) - len(added_fs), len(out_fs))
        all_sel_fids = np.concatenate([all_sel_fids, sel_fids], axis=0)

        # delaunay
        l = get_mollified_edge_length(out_vs, out_fs[all_sel_fids])
        _, add_fs = igl.intrinsic_delaunay_triangulation(l.astype('f8'), out_fs[all_sel_fids])
        out_fs[all_sel_fids] = add_fs

    # update FI, remove deleted faces
    FI = np.arange(len(fs))
    FI[out_fs[:len(fs), 0] < 0] = -1
    idx = np.where(FI >= 0)[0]
    FI[idx] = np.arange(len(idx))
    out_fs = out_fs[out_fs[:, 0] >= 0]
    return out_vs, out_fs, FI


def triangulate_refine_fair(vs, fs, hole_len_thr=10000, close_hole_fast=True, density_factor=np.sqrt(2), fair_alpha=0.05):
    # fill
    out_fs = close_holes(vs, fs, hole_len_thr, close_hole_fast)
    add_fids = np.arange(len(fs), len(out_fs))

    # refine
    nv = len(vs)
    out_vs, out_fs, FI = triangulation_refine_leipa(vs, out_fs, add_fids, density_factor)
    add_vids = np.arange(nv, len(out_vs))

    # fairing
    out_vs = mesh_fair_laplacian_energy(out_vs, out_fs, add_vids, fair_alpha)
    return out_vs, out_fs
