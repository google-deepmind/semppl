# SemPPL

This implementation provides the linear evaluation pipeline for SemPPL. The
module `eval_experiment.py` trains a linear classifier on ImageNet and evaluates
the performance of the frozen encoder/representation learnt by SemPPL on the
ImageNet test set.

The code provided in this repository is based on the BYOL code
(https://github.com/deepmind/deepmind-research/tree/master/byol).

## Setup

### Environment

To set up a Python virtual environment with the required dependencies, run:

```shell
python3 -m venv semppl_env
source semppl_env/bin/activate
pip install --upgrade pip
pip install -r semppl/requirements.txt
```

### Dataset

The code uses `tensorflow_datasets` to load the ImageNet dataset. Manual
download may be required; see
https://www.tensorflow.org/datasets/catalog/imagenet2012 for details.

## Full pipeline for linear evaluation on ImageNet

The various parts of the pipeline can be run using:

```shell
python -m semppl.main_loop \
  --worker_mode=<'train' or 'eval'> \
  --checkpoint_root=</path/to/the/checkpointing/folder> \
```

Use `--worker_mode=train` for a training job, which will load the encoder
weights from an existing checkpoint (from a pretrain experiment) located at
`<checkpoint_root>/pretrain.pkl`, and train a linear classifier on top of this
encoder. The main loop for linear evaluation runs for 100 epochs.

The training job will regularly save checkpoints under
`<checkpoint_root>/linear-eval.pkl`. You can run a second worker (using
`--worker_mode=eval`) with the same `checkpoint_root` setting to regularly load
the checkpoint and evaluate the performance of the classifier (trained by the
linear-eval `train` worker) on the test set.

Note that the config/eval.py is set-up for using the ResNet50 1x architecture.
If you want to run the code for different architectures, please change the
encoder_class in config/eval.py.

## SemPPL Checkpoints

We will update this once the GCP bucket is public.

## Citing this work

If you use this code please cite:

```
@inproceedings{bosnjak2023semppl,
    title={SemPPL: Predicting pseudo-labels for better contrastive representations},
    author={Bo{\v{s}}njak, Matko and Richemond, Pierre H and Tomasev, Nenad and Strub, Florian and Walker, Jacob C and Hill, Felix and Buesing, Lars Holger and Pascanu, Razvan and Blundell, Charles and Mitrovic, Jovana},
    booktitle={Proceedings of International Conference on Learning Representations (ICLR)},
    year={2023},
}
```

## License and disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
