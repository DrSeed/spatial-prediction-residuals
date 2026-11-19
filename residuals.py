# Where does tissue morphology fail to predict expression? Spatially cross-validated residual maps.
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def block_folds(shape, block=10, n_folds=5, seed=0):
    """Assign grid spots to folds in square blocks, so test spots aren't surrounded by training spots."""
    rng = np.random.default_rng(seed)
    by, bx = np.indices(shape)
    blocks = (by // block) * (shape[1] // block + 1) + bx // block
    fold_of_block = rng.integers(0, n_folds, blocks.max() + 1)
    return fold_of_block[blocks].ravel()


def predict_expression(morph, expr, folds):
    """Out-of-fold predictions of every gene from morphology features (ridge on pairwise feature
    products, flexible enough that leftover structure isn't just the model being too simple)."""
    pred = np.zeros_like(expr)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        model = make_pipeline(PolynomialFeatures(2), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
        model.fit(morph[tr], expr[tr])
        pred[te] = model.predict(morph[te])
    return pred


def morans_i(values, shape):
    """Moran's I on a grid with rook (up/down/left/right) neighbours."""
    z = values.reshape(shape) - values.mean()
    num = 2 * ((z[1:, :] * z[:-1, :]).sum() + (z[:, 1:] * z[:, :-1]).sum())
    n_links = 2 * ((shape[0] - 1) * shape[1] + shape[0] * (shape[1] - 1))
    return (z.size / n_links) * num / (z ** 2).sum()


def morans_test(values, shape, n_perm=199, seed=0):
    rng = np.random.default_rng(seed)
    obs = morans_i(values, shape)
    null = np.array([morans_i(rng.permutation(values), shape) for _ in range(n_perm)])
    return obs, (1 + (null >= obs).sum()) / (n_perm + 1)


def discordance_map(residuals, shape, smooth=7):
    """Per-spot mean squared standardised residual across genes, smoothed over a small window."""
    z = residuals / residuals.std(axis=0)
    return uniform_filter((z ** 2).mean(axis=1).reshape(shape), size=smooth)
