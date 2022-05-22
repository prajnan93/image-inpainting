# Image Inpainting using Generative Adversarial Network

___

### About

A pytorch implementation of the paper [Free Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589v2). The code can be install as a python package called `inpaint`. The inpaint package provides APIs to :
 - Create a Pytorch Dataset for Places365 dataset
   ```
   from inpaint.data import PlacesDataset 
   ```
 - Configure and setup Generator and Discriminator
   ```
   from inpaint.core.discriminator import PatchDiscriminator
   from inpaint.core.generator import GatedGenerator'
   ````
 - Configure and setup training, evaluation and prediction
   ```
   from inpaint.tools import Trainer, Evaluate, predict
   ```
 
 Tutorial and documentation for APIs are provided in the `examples` directory.


### Results 
<img width="1082" alt="Screenshot 2022-05-22 at 4 07 24 PM" src="https://user-images.githubusercontent.com/63877211/169713817-960b4e8c-fd6d-4bea-b36f-2050a499bd0b.png">

___
### Setup

Here's how to set up `inpaint` for local development and testing.

1. Install [Miniconda](https://conda.io/miniconda.html)

2. Clone the repo locally::

    $ git clone https://github.com/prajnan93/image-inpainting

3. Create a Conda virtual environment using the `environment.yml` file.  Install your local copy of the package into the environment::

    - $ conda env create -f environment.yml
    - $ conda activate inpaint
    - $ python setup.py develop

4. Please note this repo is not accepting any contributions.

___
### Tutorials and Documentation

- Setup the inpaint conda environment as mentioned above.
- Follow the instructions provided in the jupyter notebooks in the directory `examples`.
- Each notebook in the `examples` directory provides an example of mask visualization, training, evaluation and prediction.
- Make sure to have at least 16Gb of CUDA GPU memory for training the model. Few example training scripts with differnt model configurations are provided in the `scripts` directory.
- Please note checkpoints for prediction are not included yet (Coming Soon).  

___

### Checkpoints of Pretrained models and Predictions in Google Colab 

coming soon. 

___

### Acknowledgements

This repository acknowleges the official implementation of [DeepFillV2 Free Form Image Inpainting](https://arxiv.org/abs/1806.03589v2) and the [Places365 Dataset](http://places2.csail.mit.edu/index.html) datasets. 

And thanks to [Northeastern University Discovery HPC](https://rc.northeastern.edu/) for providing the compute support.
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

___

### License

This python package is for education and research purposes only.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
