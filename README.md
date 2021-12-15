# VAE
VAE model implemented on pytorch for various dataset

The VAE code is based on [here](https://github.com/AntixK/PyTorch-VAE). The origin code was built on pytorch_lightning, but we converted it on original pytorch.



### How to run

```
$ git clone https://github.com/DVL-Sejong/VAE
$ cd VAE
$ conda env create --file=environment.yaml
$ conda activate VAE
$ pip install -r requirements.txt
$ python run.py
```

or, simply run this

```
$ git clone https://github.com/DVL-Sejong/VAE
$ cd VAE
$ bash run.sh
```



### dataset

- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - Download data from the link, and put those under `/your_data_path/celeba/*`

