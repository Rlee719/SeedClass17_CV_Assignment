# Code for CV Assignment of SeedClass_17
***[Rlee719](https://github.com/Rlee719), [RyuGuo](https://github.com/RyuGuo), [Skylyyun](https://github.com/Skylyyun)***
## Project 1: Building an image classifier on Cifar10 dataset using image features and classifiers

- [ ] Cifar10 https://www.cs.toronto.edu/~kriz/cifar.html, 50000 training images and 10000 test images. 
- [ ] Using raw pixels as image feature
- [ ] Using kNN classifier
- [ ] Determine the best k & distance metric
- [ ] Due in the next work: source code, slides and oral presentation

把 cifar-10-batches-py 放置在 project1 目录。

## Project 2: Implementing a Softmax classifier for Cifar10 image classification

- [ ] 自己推导 softmax clf 的 analytic grad，并采用 numpy 实现其计算，用 numerical grad 来检验自己实现的正确性。
- [ ] 采用 mini-batch grad descent 进行优化，调整 batch ，learning rate，epoch 等。
- [ ] 尝试 L2，L1 正则化方法（ grad 自己推导），调整正则化项的权重。
- [ ] 注意数据的预处理，例如：像素值归一化到 0-1 ，调整参数，不用交叉验证，争取在整个训练集上训练，达到较高的测试准确率。