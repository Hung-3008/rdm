"""ROI figure template: stacked coronal slices with a label overlay (panel A) and
3-D renders of the same ROIs as points, wireframe, matte surface and specular
surface (panels B-E).

Inputs: an anatomical volume and an integer label volume on the same voxel grid,
the grid's affine (voxel -> mm), and {region name: [label ids]}. Load volumes with
`load_canonical` so that array axes are (x=R, y=A, z=S).

    anat, aff = load_canonical("T1.mgz")
    lab, _ = load_canonical("aparc+aseg.mgz")
    fig = roi_figure(anat, lab, aff, {"Hippocampus": [53], "Amygdala": [54]})
    fig.savefig("roi.png", dpi=200)

Needs scikit-image (marching cubes) and pyvista/vtk (off-screen rendering).
"""
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pyvista as pv
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy import ndimage
from skimage import measure

PALETTE = ["#3f7cc4", "#d93b2b", "#4daf4a", "#f39234",
           "#984ea3", "#f2c12e", "#17becf", "#a65628"]

# pyvista add_mesh kwargs per panel
STYLES = {
    "points": dict(style="points", point_size=2.5, opacity=0.8, lighting=False),
    "wireframe": dict(style="wireframe", line_width=0.7, opacity=0.55, lighting=False),
    "surface": dict(style="surface", smooth_shading=True,
                    ambient=0.3, diffuse=0.75, specular=0.0),
    "specular": dict(style="surface", smooth_shading=True,
                     ambient=0.2, diffuse=0.7, specular=0.7, specular_power=30),
}
# fraction of triangles removed before drawing, so single points / edges stay visible
STYLE_DECIMATE = {"points": 0.7, "wireframe": 0.7}
STYLE_TITLES = {"points": "Points", "wireframe": "Wireframe",
                "surface": "Surface", "specular": "Surface"}


def load_canonical(path):
    """Volume reoriented to RAS+, returned as (array, affine)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(img.dataobj), img.affine


def region_masks(labels, regions):
    """{name: bool mask} with mask = labels in ids."""
    return {name: np.isin(labels, ids) for name, ids in regions.items()}


def mask_to_mesh(mask, affine, sigma=1.0, smooth_iter=30, pad=4):
    """Triangle mesh (mm coordinates) of a binary mask.

    The mask is blurred with a Gaussian of `sigma` voxels, cut at 0.5 by marching
    cubes, and Taubin-smoothed for `smooth_iter` iterations.
    """
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    vol = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    if sigma > 0:
        vol = ndimage.gaussian_filter(vol, sigma)
    verts, faces, _, _ = measure.marching_cubes(vol, 0.5)
    verts = nib.affines.apply_affine(affine, verts + lo)
    mesh = pv.PolyData(verts, np.c_[np.full(len(faces), 3), faces].ravel())
    if smooth_iter:
        mesh = mesh.smooth_taubin(n_iter=smooth_iter, pass_band=0.05)
    return mesh.compute_normals(auto_orient_normals=True)


def _view_dir(azimuth, elevation):
    """Unit vector from the focal point to the camera, RAS mm.

    azimuth: degrees in the axial plane from +x (right) toward +y (anterior);
    elevation: degrees above the axial plane.
    """
    a, e = np.deg2rad(azimuth), np.deg2rad(elevation)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _crop_alpha(img, pad=4):
    ys, xs = np.nonzero(img[..., 3] > 0)
    if len(ys) == 0:
        return img
    return img[max(ys.min() - pad, 0):ys.max() + pad + 1,
               max(xs.min() - pad, 0):xs.max() + pad + 1]


def render_meshes(meshes, colors, style="specular", view=(0, 0), up=(0, 0, 1),
                  size=(900, 900), zoom=1.0, crop=True, parallel_scale=None, focal=None):
    """RGBA image (H, W, 4) of meshes drawn in one of STYLES.

    meshes / colors: {name: pv.PolyData} / {name: color or (n_points, 3) RGB in
    [0, 1]}; view: (azimuth, elevation) in degrees, see `_view_dir`.
    parallel_scale: if given, orthographic camera centred on `focal` (default the
    bounds centre) whose image height spans 2 * parallel_scale mm, with no auto-fit,
    so a point's pixel position follows from its mm coordinates.
    """
    p = pv.Plotter(off_screen=True, window_size=list(size))
    p.set_background("white")
    for name, m in meshes.items():
        c = colors[name]
        per_vertex = np.ndim(c) == 2
        if STYLE_DECIMATE.get(style) and not per_vertex:
            m = m.decimate(STYLE_DECIMATE[style])
        kw = dict(scalars=np.asarray(c), rgb=True) if per_vertex else dict(color=c)
        p.add_mesh(m, **kw, **STYLES[style])
    if STYLES[style].get("opacity", 1.0) < 1.0:
        p.enable_depth_peeling()
    bounds = np.array(pv.MultiBlock(list(meshes.values())).bounds).reshape(3, 2)
    center = bounds.mean(1) if focal is None else np.asarray(focal, float)
    p.camera_position = [center + 500 * _view_dir(*view), center, up]
    if parallel_scale is None:
        p.reset_camera()
        p.camera.zoom(zoom)
    else:
        p.enable_parallel_projection()
        p.camera.parallel_scale = parallel_scale
        p.reset_camera_clipping_range()
    img = p.screenshot(transparent_background=True, return_img=True)
    p.close()
    return _crop_alpha(img) if crop else img


def _anat_norm(anat):
    v = anat[anat > 0]
    lo, hi = (np.percentile(v, [0.5, 99.5]) if v.size else (0.0, 1.0))
    return np.clip((anat - lo) / (hi - lo + 1e-8), 0, 1)


def default_slices(masks, n=4, q=(0.2, 0.9)):
    """n coronal (y) indices spread over quantiles q of the ROI union, posterior first."""
    ys = np.argwhere(np.any(np.stack(list(masks.values())), 0))[:, 1]
    return np.round(np.quantile(ys, np.linspace(*q, n))).astype(int).tolist()


def draw_slices(fig, rect, anat, masks, colors, ys, radiological=False,
                overlap=0.32, alpha=1.0, thresh=0.03, pad=6):
    """Coronal slices at y indices `ys`, overlapping left to right inside `rect`.

    Later (more anterior when ys is increasing) slices are drawn on top. Voxels
    outside the head (filled anat > thresh) are transparent and the head outline is
    drawn in white. Masks are painted as colors with weight `alpha`.
    """
    a = _anat_norm(anat.astype(np.float32))
    head = np.any(a[:, ys, :] > thresh, axis=1)  # (x, z)
    xi, zi = np.nonzero(head)
    x0, x1 = max(xi.min() - pad, 0), min(xi.max() + pad + 1, a.shape[0])
    z0, z1 = max(zi.min() - pad, 0), min(zi.max() + pad + 1, a.shape[2])
    aspect = (z1 - z0) / (x1 - x0)

    L, B, W, H = rect
    n = len(ys)
    fig_w, fig_h = fig.get_size_inches()
    # width of one slice such that n slices with `overlap` fit in W and H
    w = min(W / (1 + (n - 1) * (1 - overlap)), H * fig_h / fig_w / aspect)
    h = w * aspect * fig_w / fig_h
    for k, y in enumerate(ys):
        sl = a[x0:x1, y, z0:z1].T[::-1]  # rows = superior -> inferior
        rgb = np.repeat(sl[..., None], 3, -1)
        for name, m in masks.items():
            mk = m[x0:x1, y, z0:z1].T[::-1]
            rgb[mk] = (1 - alpha) * rgb[mk] + alpha * np.array(to_rgb(colors[name]))
        hd = ndimage.binary_fill_holes(ndimage.binary_closing(sl > thresh, iterations=6))
        lab_cc, ncc = ndimage.label(hd)
        if ncc > 1:
            hd = lab_cc == 1 + np.argmax(ndimage.sum(hd, lab_cc, range(1, ncc + 1)))
        rgba = np.concatenate([rgb, hd[..., None].astype(float)], -1)
        if radiological:
            rgba = rgba[:, ::-1]
        ax = fig.add_axes([L + k * w * (1 - overlap), B + (H - h) / 2, w, h], zorder=k)
        ax.imshow(rgba, interpolation="bilinear")
        ax.contour(rgba[..., 3], levels=[0.5], colors="white", linewidths=2.0)
        ax.axis("off")
    return [L, B + (H - h) / 2, w * (1 + (n - 1) * (1 - overlap)), h]


def roi_figure(anat, labels, affine, regions, colors=None, ys=None,
               slice_q=(0.2, 0.9), view=(0, 0), up=(0, 0, 1), radiological=False,
               styles=("points", "wireframe", "surface", "specular"),
               mesh_kw=None, figsize=(9.0, 5.4), title=None):
    """Panel A (stacked coronal slices + legend) and panels B.. (3-D styles).

    anat, labels: arrays on one RAS+ grid; affine: voxel -> mm of that grid;
    regions: {name: [label ids]}; colors: {name: color} (default PALETTE);
    ys: coronal slice indices (default `default_slices` at quantiles slice_q
    of the ROI union along y); view/up: camera, see
    `render_meshes`. Returns the matplotlib Figure.
    """
    colors = colors or {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(regions)}
    masks = {n: m for n, m in region_masks(labels, regions).items() if m.any()}
    ys = default_slices(masks, q=slice_q) if ys is None else ys
    meshes = {n: mask_to_mesh(m, affine, **(mesh_kw or {})) for n, m in masks.items()}

    fig = plt.figure(figsize=figsize, facecolor="white")
    box = draw_slices(fig, [0.06, 0.53, 0.62, 0.41], anat, masks, colors, ys,
                      radiological=radiological)
    fig.text(0.02, 0.95, "A", fontsize=15, fontweight="bold", va="top")
    fig.legend(handles=[Patch(color=colors[n], label=n) for n in masks],
               loc="center left", bbox_to_anchor=(box[0] + box[2] + 0.03, 0.75),
               frameon=False, fontsize=11, handlelength=1.1, handleheight=1.1,
               prop={"weight": "bold", "size": 11})

    n = len(styles)
    for k, st in enumerate(styles):
        img = render_meshes(meshes, colors, st, view=view, up=up)
        x = 0.02 + k * 0.96 / n
        ax = fig.add_axes([x, 0.1, 0.96 / n - 0.01, 0.4])
        ax.imshow(img)
        ax.axis("off")
        fig.text(x + 0.005, 0.5, "BCDEFGH"[k], fontsize=15, fontweight="bold", va="top")
        fig.text(x + 0.48 / n, 0.07, STYLE_TITLES[st], ha="center", va="top",
                 fontsize=16, fontweight="bold")
        if st == "specular":
            fig.text(x + 0.48 / n, 0.025, "with specular", ha="center", va="top", fontsize=10)
    if title:
        fig.suptitle(title, x=0.98, ha="right", fontsize=10, color="0.35")
    return fig
