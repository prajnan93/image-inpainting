# Image Inpainting

___

### About


___
### Setup

Here's how to set up `inpaint` for local development and testing.

1. Install [Miniconda](https://conda.io/miniconda.html)

2. Clone the repo locally::

    $ git clone https://github.com/prajnan93/image-inpainting

3. Create a Conda virtual environment using the `environment.yml` file.  Install your local copy of the package into the environment::

    $ conda env create -f environment.yml
    $ conda activate inpaint
    $ python setup.py develop

4. Please note this repo is not accepting any contributions.

___
### Tutorials and Documentation

- Setup the inpaint conda environment as mentioned above.
- Follow the instructions provided in the jupyter notebooks in the directory `examples`.
- Each notebook in the `examples` directory provides an example of dataloading, training, evaluation and prediction.
- Make sure to have at least 16Gb of CUDA GPU memory for training the model.
- Please note checkpoints for prediction are not included yet (Coming Soon).  

___

### Checkpoints of Pretrained models and Predictions in Google Colab 

coming soon.

___

### Acknowledgements


___

### Citations

```
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}

@article{yu2018free,
  title={Free-Form Image Inpainting with Gated Convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}

@article{zhou2017places,
  title={Places: A 10 million Image Database for Scene Recognition},
  author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2017},
  publisher={IEEE}
}
```

___

### References
- [Free-Form Image Inpainting with Gated Convolution](https://github.com/JiahuiYu/generative_inpainting)

- [GatedConvolution_pytorch](https://github.com/avalonstrel/GatedConvolution_pytorch)

- [DeepFillv2_Pytorch](https://github.com/csqiangwen/DeepFillv2_Pytorch)

- [Places365 Dataset](http://places2.csail.mit.edu/index.html)

