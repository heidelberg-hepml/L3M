import numpy as np
import os
import torch

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec

from src.utils.datasets import get_preprocessing_layers

@torch.inference_mode()
def evaluate_model(model, eval_dataset, out_dir, label_names, collate_fct):
    model.eval()

    labels = []
    preds = []
    sigmas = []
    for data in eval_dataset:
        x = collate_fct([data])

        # predict
        for k, val in x.items():
            x[k] = val.to(model.device)
        y = x['labels'].to(model.device)

        output = model(**x, use_cache=False, return_dict = True)
        pred_params = output.logits[0].detach().cpu()
        _sigma = output.sigmas[0].detach().cpu()

        sigmas.append(_sigma.numpy())
        preds.append(pred_params.numpy())
        labels.append(y.cpu().numpy())

    # stack results
    labels = np.vstack(labels)
    preds = np.vstack(preds)
    sigmas = np.vstack(sigmas)

    # save results
    savearrs = [labels, preds]
    add_kwargs = {}
    add_kwargs["sigmas_inv"] = sigmas
    
    savepath = os.path.join(out_dir, "label_pred_pairs.npz")
    np.savez(savepath, label_pred_pairs=np.stack(savearrs, axis=-1), **add_kwargs)

def plot(out_dir, label_names, n_samples=50):
    print(f"Sampling {n_samples} per prediction")
    data = np.load(os.path.join(out_dir, "label_pred_pairs.npz"))
    res_data = data["label_pred_pairs"]
    sigmas = data["sigmas_inv"].reshape(-1, 6, 6)

    processing = get_preprocessing_layers(label_names)

    res_labels = []
    res_preds = []
    print(f"Res data {res_data.shape}")
    print(f"sigmas: {sigmas.shape}")
    for i in range(res_data.shape[0]):
        labels, preds = res_data[i, :, :].T
        assert len(labels) == len(label_names)

        labels = torch.from_numpy(labels)
        preds = torch.from_numpy(preds)
        
        sigma = torch.from_numpy(sigmas[i])
        normal = torch.distributions.MultivariateNormal(loc=preds, precision_matrix=sigma)
        for i in range(n_samples):
            sample = normal.sample()
            sample = sample.clamp(min=0., max=1.)
            res_labels.append(processing.reverse(labels).numpy())
            res_preds.append(processing.reverse(sample).numpy())

    labels = np.asarray(res_labels)
    preds = np.asarray(res_preds)

    print(f"labels: {labels.shape}")
    print(f"preds: {preds.shape}")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}" r"\usepackage[bitstream-charter]{mathdesign}"
    )

    savename = f"regression.pdf"
    savepath = os.path.join(out_dir, savename)
    if os.path.exists(savepath):
        old_dir = os.path.join(out_dir, "old_plots")
        os.makedirs(old_dir, exist_ok=True)
        os.rename(savepath, os.path.join(old_dir, savename))

    # marker settings
    ref_kwargs = {
        "color": "#171717",
        "ls": "-",
        "lw": 0.5,
        "alpha": 0.8
    }
    err_kwargs = {
        "fmt": "o",
        "ms": 2,
        "elinewidth": 2.0,
    }

    ratio_s = 10
    param_names = label_names

    # create plots
    with PdfPages(savepath) as pdf:

        fig = plt.figure(constrained_layout=True, figsize=(10, 8))
        subfigs = fig.subfigures(2, 3)

        # iterate over individual parameters
        for i, j in np.ndindex((2,3)):
            k = 3 * i + j
            if k >= len(label_names):
                break

            l3m_preds = preds[:, k]
            l3m_labels = labels[:, k]

            print(f"l3m_preds: {l3m_preds.shape}")
            print(f"l3m_labels: {l3m_labels.shape}")

            param_name = param_names[k]

            lo, hi = np.min(l3m_labels), np.max(l3m_labels)
            print(f"lo: {lo}")
            print(f"hi: {hi}")

            subfig = subfigs[i][j]
            grid = gridspec.GridSpec(
                2, 1, figure=subfig, height_ratios=[5, 1.5], hspace=0.05
            )
            main_ax = plt.subplot(grid[0])
            ratio_ax = plt.subplot(grid[1])

            # digitize data
            num_bins = 10
            bins = np.linspace(lo, hi, num_bins + 1)
            bin_centers = (bins[1:] + bins[:-1]) / 2

            l3m_bin_idcs= np.digitize(l3m_labels, bins)
            l3m_partitions = [l3m_preds[l3m_bin_idcs == i + 1] for i in range(num_bins)]
            l3m_mares = np.abs(l3m_labels - l3m_preds) / (hi - lo)
            l3m_errs = list(map(np.std, l3m_partitions))
            l3m_mare_partitions = [l3m_mares[l3m_bin_idcs == i + 1] for i in range(num_bins)]

            main_ax.errorbar(
                bin_centers,
                list(map(np.mean, l3m_partitions)),
                yerr=l3m_errs,
                **err_kwargs,
                color="black"
            )

            ratio_ax.scatter(
                bin_centers,
                list(map(np.mean, l3m_mare_partitions)),
                s=ratio_s,
                color="black"
            )

            # fill main axis
            pad = 0.04 * (hi - lo)
            main_ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], **ref_kwargs)
            main_ax.set_title(param_name, fontsize=18) # fontsize=14)
            if j == 0:
                main_ax.set_ylabel("Network", fontsize=16) #13)
                ratio_ax.set_ylabel(
                    r"$\left|\frac{\text{Net}\,-\,\text{True}}{\text{Max} - \text{Min}}\right|$",
                    fontsize=16,
                )
            if i == 1:
                ratio_ax.set_xlabel("Truth", fontsize=16)

            # axis limits
            main_ax.set_xlim([lo - pad, hi + pad])
            main_ax.set_ylim([lo - pad, hi + pad])
            ratio_ax.set_xlim(*main_ax.get_xlim())
            ratio_ax.set_ylim(bottom=0.)

            # clean
            main_ax.set_xticklabels([])

        fig.suptitle("Regression results", fontsize=20, ha='left', x=0.1)
        pdf.savefig(fig, bbox_inches="tight")


