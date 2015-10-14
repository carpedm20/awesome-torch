# Awesome Torch

A curated list of awesome Torch tutorials, projects and communities.


## Table of Contents

- [Tutorials](#tutorials)
- [Model Zoo](#model-zoo)
    - [Recurrent Networks](#recurrent-networks)
    - [Convolutional Networks](#convolutional-networks)
    - [ETC](#model-zoo-etc)
- [Libraries](#libraries)
    - [Model related](#model-related)
    - [GPU related](#gpu-related)
    - [IDE related](#ide-related)
    - [ETC](#libraries-etc)
- [Links](#links)


## Tutorials

- [Applied Deep Learning for Computer Vision with Torch](http://torch.ch/docs/cvpr15.html) CVPR15 Tutorial [[Slides]](https://github.com/soumith/cvpr2015/blob/master/cvpr-torch.pdf)
    - [Deep Learning with Torch - A 60-minute blitz](https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb)
    - [NNGraph - graph style neural networks](https://github.com/soumith/cvpr2015/blob/master/NNGraph%20Tutorial.ipynb)
    - [Character-level Recurrent networks. An introduction to real-world nngraph RNN training](https://github.com/soumith/cvpr2015/blob/master/Char-RNN.ipynb)
    - [Deep-Q Reinforcement learning to play Atari games](https://github.com/soumith/cvpr2015/blob/master/DQN%20Training%20iTorch.ipynb)
- [Machine Learning with Torch](http://code.madbits.com/wiki/doku.php) for IPAM Summer School on Deep Learning. [[Code]](https://github.com/torch/tutorials)
- [Oxford Computer Science - Machine Learning 2015](https://github.com/oxford-cs-ml-2015)
- [Implementing LSTMs with nngraph](http://apaszke.github.io/lstm-explained.html)
- [Community Wiki (Cheatseet) for Torch](https://github.com/torch/torch7/wiki/Cheatsheet)
- [Demos & Turorials for Torch](https://github.com/torch/demos)
- [Learn Lua in 15 Minutes](http://tylerneylon.com/a/learn-lua/)



## Model Zoo

Codes and related articles. `(#)` means authors of code and paper are different.

### Recurrent Networks

- [SCRNN (Structurally Constrained Recurrent Neural Network)](https://github.com/facebook/SCRNNs)
    - Tomas Mikolov, Armand Joulin, Sumit Chopra, Michael Mathieu, Marc'Aurelio Ranzato, *Learning Longer Memory in Recurrent Neural Networks*, arXiv:1406.1078 [[Paper]](http://arxiv.org/abs/1412.7753)
- [Tree-LSTM (Tree-structured Long Short-Term Memory networks)](https://github.com/stanfordnlp/treelstm)
    - Kai Sheng Tai, Richard Socher, Christopher D. Manning, *Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks*, ACL 2015 [[Paper]](http://arxiv.org/abs/1503.00075)
- [LSTM language model with CNN over characters](https://github.com/yoonkim/lstm-char-cnn)
    - Yoon Kim, Yacine Jernite, David Sontag, Alexander M. Rush, *Character-Aware Neural Language Models*, arXiv:1508.06615 [[Paper]](http://arxiv.org/abs/1508.06615)
- [LSTM, GRU, RNN for character-level language (char-rnn)](https://github.com/karpathy/char-rnn)
    - Andrej Karpathy, Justin Johnson, Li Fei-Fei, *Visualizing and Understanding Recurrent Networks*, arXiv:1506.02078 [[Paper]](http://arxiv.org/abs/1506.02078)
- [LSTM for word-level language model](https://github.com/wojzaremba/lstm)
    - Wojciech Zaremba, Ilya Sutskever, Oriol Vinyal, *Recurrent Neural Network Regularization*, arXiv:1409.2329 [[Paper]](http://arxiv.org/abs/1409.2329)
- [LSTM](https://github.com/wojciechz/learning_to_execute)
    - Wojciech Zaremba, Ilya Sutskever, *Learning to Execute*, arXiv:1410.4615 [[Paper]](http://arxiv.org/abs/1410.4615)
- [Grid LSTM](https://github.com/sherjilozair/grid-lstm)
    - (#) Nal Kalchbrenner, Ivo Danihelka, Alex Graves, *Grid Long Short-Term Memory*, arXiv:1507.01526, [[Paper]](http://arxiv.org/abs/1507.01526)
- [Recurrent Visual Attention Model](https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua)
    - (#) Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu, *Recurrent Models of Visual Attention*, NIPS 2014 [[Paper]](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention)
- [DRAW (Deep Recurrent Attentive Writer)](https://github.com/vivanov879/draw)
    - (#) Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, Daan Wierstra, *DRAW: A Recurrent Neural Network For Image Generation*, arXiv:1502.04623 [[Paper]](http://arxiv.org/abs/1502.04623)

### Convolutional Networks

- [Crepe (Character-level Convolutional Networks for Text Classification)](https://github.com/zhangxiangxiao/Crepe)
    - Xiang Zhang, Junbo Zhao, Yann LeCun. *Character-level Convolutional Networks for Text Classification*, NIPS 2015 [[Paper]](http://arxiv.org/abs/1509.01626)
- [OpenFace (Face recognition with Google's FaceNet deep neural network)](https://github.com/cmusatyalab/openface)
    - (#) Florian Schroff, Dmitry Kalenichenko, James Philbin, *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR 2015 [[Paper]](http://arxiv.org/abs/1503.03832)
- [Neural Style](https://github.com/jcjohnson/neural-style), [Neural Art](https://github.com/kaishengtai/neuralart)
    - (#) Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, *A Neural Algorithm of Artistic Style*, arXiv:1508.06576 [[Paper]](http://arxiv.org/abs/1508.06576)
- [Overfeat](https://github.com/jhjin/overfeat-torch)
    - (#) Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, Yann LeCun, *OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks*, arXiv:1312.6229 [[Paper]](http://arxiv.org/abs/1312.6229)
- [Alexnet, Overfeat, VGG in Torch on multiple GPUs over ImageNet](https://github.com/soumith/imagenet-multiGPU.torch)

<a name="model-zoo-etc" />
### ETC

- [Neural Attention Model for Abstractive Summarization](https://github.com/facebook/NAMAS)
    - Alexander M. Rush, Sumit Chopra, Jason Weston, *A Neural Attention Model for Abstractive Summarization*, EMNLP 2015 [[Paper]](http://arxiv.org/abs/1509.00685)
- [Memory Networks](https://github.com/facebook/MemNN)
    - Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus, *End-To-End Memory Networks*, arXiv:1503.08895, [[Paper]](http://arxiv.org/abs/1503.08895)
- [Neural Turing Machine](https://github.com/kaishengtai/torch-ntm)
    - Alex Graves, Greg Wayne, Ivo Danihelka, *Neural Turing Machines*, arXiv:1410.5401 [[Paper]](http://arxiv.org/abs/1410.5401)
- [Deep Q-network](https://sites.google.com/a/deepmind.com/dqn/), [DeepMind-Atari-Deep-Q-Learner](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner)
    - Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, Demis Hassabis, *Human-Level Control through Deep Reinforcement Learning*, Nature, [[Paper]](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
- [Eyescream (Natural Image Generation using ConvNets)](https://github.com/facebook/eyescream)
    - Emily Denton, Soumith Chintala, Arthur Szlam, Rob Fergus, *Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks*, arXiv:1506.05751 [[Paper]](http://arxiv.org/abs/1506.05751)
- [BNN (Bilingual Neural Networks) with LBL and CNN](https://bitbucket.org/ketran/morphbinn)
    - Ke Tran, Arianna Bisazza, Christof Monz, *Word Translation Prediction for Morphologically Rich Languages with Bilingual Neural Networks*, EMNLP 2014 [[Paper]](http://emnlp2014.org/papers/pdf/EMNLP2014175.pdf)
- [DSSM (Deep Structured Semantic Model)](https://github.com/jiasenlu/CDSSM)
    - (#) Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, Larry Heck, *Learning Deep Structured Semantic Models for Web Search using Clickthrough Data*, CIKM 2013 [[Paper]](http://research.microsoft.com/apps/pubs/default.aspx?id=198202)
- [TripletNet](https://github.com/eladhoffer/TripletNet)
    - (#) Elad Hoffer, Nir Ailon, *Deep metric learning using Triplet network*, arXiv:1412.6622 [[Paper]](http://arxiv.org/abs/1412.6622)
- [Word2Vec](https://github.com/yoonkim/word2vec_torch)
    - (#) Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean, *Efficient Estimation of Word Representations in Vector Space*, ICLR 2013 [[Paper]](http://arxiv.org/abs/1301.3781)
- [TripletLoss (used in Google's FaceNet)](https://github.com/Atcold/torch-TripletEmbedding)
    - (#) Florian Schroff, Dmitry Kalenichenko, James Philbin, *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR 2015 [[Paper]](http://arxiv.org/abs/1503.03832)


## Libraries

### Model related

- nn : an easy and modular way to build and train simple or complex neural networks [[Code]](https://github.com/torch/nn) [[Documentation]](http://nn.readthedocs.org/en/rtd/index.html)
- rnn : Recurrent Neural Network library [[Code]](https://github.com/Element-Research/rnn)
- optim : A numeric optimization package for Torch [[Code]](https://github.com/torch/optim)
- dp : a deep learning library designed for streamlining research and development [[Code]](https://github.com/nicholas-leonard/dp) [[Documentation]](http://dp.readthedocs.org/en/latest/#tutorials-and-examples)
- nngraph : provides graphical computation for *nn* library [[Code]](https://github.com/torch/nngraph) [[Oxford Introduction]](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/practicals/practical5.pdf)

### GPU related

- cutorch : A CUDA backend for Torch [[Code]](https://github.com/torch/cutorch)
- cudnn : Torch FFI bindings for NVIDIA CuDNN [[Code]](https://github.com/soumith/cudnn.torch)
- fbcunn : Facebook's extensions to torch/cunn [[Code]](https://github.com/facebook/fbcunn) [[Documentation]](https://facebook.github.io/fbcunn/fbcunn/index.html)

### IDE related

- iTorch : IPython kernel for Torch with visualization and plotting [[Code]](https://github.com/facebook/iTorch)
- zbs-torch : A lightweight Lua-based IDE for Lua with code completion, syntax highlighting, live coding, remote debugger, and code analyzer [[Code]](https://github.com/soumith/zbs-torch)

<a name="libraries-etc" />
### ETC

- fblualib : Facebook libraries and utilities for Lua [[Code]](https://github.com/facebook/fblualib)
- loadcaffe : Load Caffe networks in Torch [[Code]](https://github.com/szagoruyko/loadcaffe)
- torch-android : Torch for Android [[Code]](https://github.com/soumith/torch-android)


## Links

- [Google Groups for torch](https://groups.google.com/forum/#!forum/torch7)
- [Gitter Chat](https://gitter.im/torch/torch7)
