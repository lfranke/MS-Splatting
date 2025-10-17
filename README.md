# 🌈 Multi-Spectral Gaussian Splatting 🌈

This repository contains the **official unified codebase** for our work on Multi-Spectral Gaussian Splatting based on the papers [Towards Integrating Multi-Spectral Imaging with Gaussian Splatting](https://arxiv.org/abs/2506.03407) and [Multi-Spectral Gaussian Splatting with Neural Color Representation](https://arxiv.org/abs/2506.03407).  

📌 The repo supports **two papers**, which share the same implementation. The code will be released **soon™** in this repository.

## 📚 Papers & Links

### **Towards Integrating Multi-Spectral Imaging with Gaussian Splatting (VMV 2025)**  
[Josef Grün](https://j-gruen.github.io), [Lukas Meyer](https://meyerls.github.io), [Maximilian Weiherer](https://mweiherer.github.io), [Bernhard Egger](https://eggerbernhard.ch), [Marc Stamminger](https://scholar.google.com), [Linus Franke](https://lfranke.github.io)

[![ArXiv](https://img.shields.io/badge/arXiv-2506.03407-b31b1b.svg)](https://arxiv.org/abs/2509.00989) [![Website](https://img.shields.io/badge/Website-🌐-4cafef)](https://meyerls.github.io/towards_multi_spec_splat/) [![Conference](https://img.shields.io/badge/Conference-VMV%202025-8a2be2)](https://www.vmv2025.fau.de/) [![YouTube](https://img.shields.io/badge/Video-YouTube-red?logo=youtube)](https://youtu.be/okqMAbUzBaE)


### **Multi-Spectral Gaussian Splatting with Neural Color Representation (MS-Splatting, ArXiv 2025)**  
[Lukas Meyer](https://meyerls.github.io), [Josef Grün](https://j-gruen.github.io), [Maximilian Weiherer](https://mweiherer.github.io), [Bernhard Egger](https://eggerbernhard.ch), [Marc Stamminger](https://scholar.google.com), [Linus Franke](https://lfranke.github.io)

[![ArXiv](https://img.shields.io/badge/arXiv-2508.14443-b31b1b.svg)](https://arxiv.org/abs/2506.03407) [![Website](https://img.shields.io/badge/Website-🌐-4cafef)](https://meyerls.github.io/ms_splatting) [![YouTube](https://img.shields.io/badge/Video-YouTube-red?logo=youtube)](https://youtu.be/5AQRJ7Ns9q0)


## 📅 Timeline

- **Winter 2025/26** – **Code release** (planned within the next weeks after paper publications)  
- **31. August 2025** – *Towards Multi-Spectral GS* uploaded to **arXiv**  
- **3. June 2025** – *MS-Splatting* uploaded to **arXiv**  


## 📖 Citation

If you use this repository, please cite the corresponding paper(s):

```bibtex
@inproceedings{gruen2025towards_msplatting,
  title     = {Towards Integrating Multi-Spectral Imaging with Gaussian Splatting},
  author    = {Grün, Josef and Meyer, Lukas and Weiherer, Maximilian and Egger, Bernhard and Stamminger, Marc and Franke, Linus},
  booktitle = {Proceedings of VMV 2025},
  year      = {2025},
  month     = {June},
  url       = {https://meyerls.github.io/towards_multi_spec_splat}
}

@article{meyer2025msplatting,
  title   = {Multi-Spectral Gaussian Splatting with Neural Color Representation},
  author  = {Meyer, Lukas and Grün, Josef and Weiherer, Maximilian and Egger, Bernhard and Stamminger, Marc and Franke, Linus},
  journal = {arXiv preprint arXiv:2508.14443},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.14443}
}
```

## 🚀 Stay Tuned

The **code, datasets, and usage instructions** will be released shortly.  
Follow this repository for updates ✨


# Code

## Installation

The following prerequisites are needed:

- python >= 3.8 (tested with 3.8.20)
- nerfstudio (tested with 1.1.5, see `pyproject.toml` and https://docs.nerf.studio/quickstart/installation.html)
- gsplat (tested with 1.4.0, see `pyproject.toml`)

Then download and install the MS-Splatting repository:

```
git clone https://github.com/j-gruen/MS-Splatting.git
cd MS-Splatting
pip install -e .
ns-install-cli
```

## Usage

#### Prepare Dataset

To prepare a multi-spectral dataset for training, run:

```
ns-process-mm-data images --data <DATA_PATH> --output-dir <OUTPUT_PATH>
```

This command uses `colmap` (has to be installed) to generate camera poses for all images. `<DATA_PATH>` must point to a directory containing the dataset images sorted into sub-folders for every multi-spectral image channel, for example:

```
data
├─ RGB
│  ├─ image1.png
│  ├─ image2.png
│  └─ ...
├─ MS_NIR
│  ├─ image1.tiff
│  ├─ image2.tiff
│  └─ ...
└─ ...
```

For more information, run `ns-process-mm-data images --help`. 


The output of `ns-process-mm-data` adheres to the standard nerfstudio `transforms.json`, with the addition of the `mm_channel` property for every frame in the dataset, and an optional `sparse_pc.ply` pointcloud for Gaussian initialization. For more information and examples look at `mmsplat_dataset.py` and our datasets.


#### Training

To train a multi-spectral model, `ns-train` can be used. Example:

```
ns-train mmsplat                 \
  --data <DATASET_PATH>        \
    --output-dir <OUTPUT_PATH>   \
    --max-num-iterations 120000  \
    mmsplat-dataparser           \
    --downscale-factor 0
```

For more information and available parameters, run `ns-train mmsplat --help` or look at the configurations in `mmsplat_model.py`, `mmsplat_datamanager.py` and `mmsplat_dataparser.py`.

To generate evaluation metrics and images, run

```
ns-eval
  --load-config <PATH/TO/config.yml>
  --output-path <OUTPUT_PATH/eval.json>
  --render-output-path <OUTPUT_IMAGES_PATH>
```

where `--load-config` points to the `config.yml` generated by ns-train.


Note that the default parameters of ns-train are not best out-of-the-box. Here is an example of the parameters used in the Joint-Optimized strategy from our VMV 2025 Paper:

```
ns-train mmsplat                                                               \
  --data <DATASET_PATH>                                                        \
  --output-dir <OUTPUT_PATH>                                                   \
  --max-num-iterations 120000                                                  \
  --pipeline.model.refine-every 300                                            \
  --pipeline.model.densification-strategy max_average                          \
  --pipeline.model.stop-split-at 60000                                         \
  --pipeline.model.pos-optim-delay-channels D 500 MS_G MS_R MS_RE MS_NIR 32000 \
  --pipeline.model.opacity-optim-delay-channels MS_G MS_R MS_RE MS_NIR 32000   \
  --pipeline.datamanager.delay-channels MS_G MS_R MS_RE MS_NIR 30000           \
  --pipeline.datamanager.channel-size D 3 MS_G MS_R MS_RE MS_NIR 1             \
  --pipeline.model.use-sh-channels D MS_G MS_R MS_RE MS_NIR                    \
  --pipeline.model.sh-degree 3                                                 \
  --pipeline.model.densification-pause-iterations 29000 32001                  \
  mmsplat-dataparser                                                           \
  --downscale-factor 0                                                         \
  --eval-mode json-list
```


#### Utility

A few more utilities like rendering and exporting are available. See the `[project.scripts]` section in `pyproject.toml` for more info.

