<p align="center">
  <h1 align="center">Collaborative Learning of Scattering and Deep Features for SAR Target Recognition with Noisy Labels (TRS' 2026)</h1>
  <p align="center">
    <a href="https://github.com/fuyimin96"><strong>Yimin Fu</strong></a>&nbsp;&nbsp;
    <strong>Zhunga Liu</strong></a>&nbsp;&nbsp;
    <strong>Dongxiu Guo</strong></a>&nbsp;&nbsp;
    <strong>Longfei Wang</strong></a>
  </p>
  <br>

Pytorch implementation for "[**Collaborative Learning of Scattering and Deep Features for SAR Target Recognition with Noisy Labels**](https://arxiv.org/pdf/2508.07656)"
<p align="center">
    <img src=./noise_saratr.png width="520">
</p>

## Requirements

To run this code, you'll need the following dependencies:

- Python 3.8
- Pytorch 1.8
- torchvision
- numpy

## Datasets
You can download the dataset for [EOC-2](https://github.com/YeRen123455/Infrared-Small-Target-Detection) and move them to `./dataset`.

## Run The Code

You can train and test the model using the following command:

```bash
# Training and test
python clsdf_train.py
```

## Citation
If you find our work and this repository useful. Please consider giving a star :star: and citation.
```bibtex
@article{fu2026clsdf,
  title={Collaborative Learning of Scattering and Deep Features for SAR Target Recognition with Noisy Labels},
  author={Fu, Yimin and Liu, Zhunga and Guo, Dongxiu and Wang, Longfei},
  journal={IEEE Transactions on Radar Systems},
  year={2026},
  publisher={IEEE}
}
```
