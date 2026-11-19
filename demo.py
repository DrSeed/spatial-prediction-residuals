# Self-contained demo: a tumour section whose morphology predicts most genes, except in a hidden niche.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from residuals import block_folds, predict_expression, morans_test, discordance_map

SHAPE = (70, 70)
TYPES = ["tumour", "stroma", "immune", "macrophage"]


def smooth_field(rng, sigma):
    f = gaussian_filter(rng.normal(size=SHAPE), sigma)
    return (f - f.mean()) / f.std()


def simulate(seed=4):
    rng = np.random.default_rng(seed)
    logits = np.stack([smooth_field(rng, 6) * 2.0 + b for b in [0.8, 0.3, -0.5, -0.8]], axis=-1)
    comp = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)           # cell-type mix per spot
    yy, xx = np.indices(SHAPE)
    niche = np.exp(-(((yy - 22) / 9) ** 2 + ((xx - 48) / 12) ** 2))        # hidden niche, e.g. hypoxic
    genes = [f"{t}_marker_{i}" for t in TYPES for i in range(6)] + [f"niche_gene_{i}" for i in range(8)] + \
            [f"housekeeping_{i}" for i in range(8)]
    G = len(genes)
    profile = rng.lognormal(0, 0.3, (len(TYPES), G))
    for t in range(len(TYPES)):
        profile[t, t * 6:(t + 1) * 6] *= 6
    C = comp.reshape(-1, len(TYPES))
    expr = C @ profile
    niche_effect = np.zeros(G)
    niche_effect[24:32] = 2.5
    expr += (niche.ravel() * C[:, 0])[:, None] * niche_effect[None, :]       # niche acts on tumour cells only
    expr = np.log1p(rng.poisson(expr * 8) / 8)
    # morphology features a vision model would see: composition (noisily) plus texture features
    morph = np.hstack([C + rng.normal(0, 0.05, C.shape), rng.normal(0, 1, (C.shape[0], 6))])
    return expr, morph, comp, niche, genes


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    expr, morph, comp, niche, genes = simulate()
    folds = block_folds(SHAPE)
    pred = predict_expression(morph, expr, folds)
    res = expr - pred

    rows = []
    for g, name in enumerate(genes):
        r2 = 1 - ((expr[:, g] - pred[:, g]) ** 2).sum() / ((expr[:, g] - expr[:, g].mean()) ** 2).sum()
        i_expr, _ = morans_test(expr[:, g], SHAPE, n_perm=49)
        i_res, p_res = morans_test(res[:, g], SHAPE, n_perm=199, seed=g)
        rows.append({"gene": name, "group": name.split("_")[0], "cv_r2": r2, "morans_i_expression": i_expr,
                     "morans_i_residual": i_res, "residual_perm_p": p_res})
    genes_df = pd.DataFrame(rows).round(4)
    genes_df.to_csv("results/per_gene.csv", index=False)

    disc = discordance_map(res, SHAPE)
    hot = disc > np.quantile(disc, 0.95)
    in_niche = niche > 0.5
    tum = comp[..., 0]
    summary = genes_df.groupby("group")[["cv_r2", "morans_i_residual"]].mean().round(3)
    summary["frac_residual_I_significant"] = genes_df.groupby("group")["residual_perm_p"].apply(lambda p: (p < 0.01).mean()).round(2)
    summary.loc["niche AUROC: residual discordance", "cv_r2"] = np.nan
    summary["value"] = np.nan
    auc_disc = roc_auc_score(in_niche.ravel(), disc.ravel())
    auc_morph = max(roc_auc_score(in_niche.ravel(), comp[..., k].ravel()) for k in range(comp.shape[-1]))
    summary.loc["niche AUROC: residual discordance", "value"] = round(auc_disc, 3)
    summary.loc["niche AUROC: best single cell-type fraction", "value"] = round(auc_morph, 3)
    summary.loc["hotspot (top 5%) spots inside niche", "value"] = round((hot & in_niche).sum() / hot.sum(), 3)
    summary.to_csv("results/summary.csv")
    tumour_rich = tum > np.median(tum)
    adj = pd.DataFrame({"region": ["hotspot", "elsewhere"],
                        "mean_tumour_fraction": [tum[hot].mean(), tum[~hot].mean()],
                        "niche_gene_residual_tumour_rich_spots": [res[:, 24:32].mean(1).reshape(SHAPE)[hot & tumour_rich].mean(),
                                                                   res[:, 24:32].mean(1).reshape(SHAPE)[~hot & tumour_rich].mean()]}).round(3)
    adj.to_csv("results/hotspot_composition.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    im = axes[0].imshow(tum, cmap="RdPu")
    axes[0].contour(niche, levels=[0.5], colors="k", linestyles="--")
    axes[0].set_title("What morphology sees: tumour fraction\n(dashed: hidden niche, invisible in the image)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    im = axes[1].imshow(disc, cmap="magma")
    axes[1].contour(niche, levels=[0.5], colors="w", linestyles="--")
    axes[1].contour(hot, levels=[0.5], colors="#5ad1ff", linewidths=1)
    axes[1].set_title(f"Residual 'discordance' map (blue: top 5%)\nfinds the niche: AUROC {auc_disc:.2f} vs {auc_morph:.2f} from any cell-type fraction")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for a in axes[:2]:
        a.set_xticks([]); a.set_yticks([])

    colours = {"tumour": "#e377c2", "stroma": "#8c564b", "immune": "#2ca02c", "macrophage": "#ff7f0e",
               "niche": "#1f77b4", "housekeeping": "#7f7f7f"}
    for grp, sub in genes_df.groupby("group"):
        axes[2].scatter(sub["morans_i_expression"], sub["cv_r2"], s=20 + 400 * sub["morans_i_residual"].clip(0),
                        color=colours[grp], alpha=0.8, label=grp)
    axes[2].set_xlabel("Moran's I of expression (spatial autocorrelation)")
    axes[2].set_ylabel("cross-validated R^2 from morphology")
    axes[2].set_title("Predictable genes are the spatially smooth ones;\nniche genes keep structure in their residuals (bigger dots)")
    axes[2].legend(fontsize=8, markerscale=0.6)
    fig.tight_layout()
    fig.savefig("figures/demo.png", dpi=120)
    print(summary.to_string())
    print(adj.to_string(index=False))


if __name__ == "__main__":
    main()
