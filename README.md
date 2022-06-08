# Edge Detection Toolbox
This toolbox was created in the course of my master thesis: Machine Learned Edges for Edge-Based Visual Odometry.

## Abstract: Machine Learned Edges for Edge-Based Visual Odometry

In this master's thesis we propose a new way of generating training data automatically for state-of-the-art machine learning based edge detectors to determine the most stable edges for edge based Visual Odometry systems.

Typically learning models need a huge amount of training data which is often done by humans. As a result the acquisition of labelled or annotated ground truth is a time consuming and cumbersome procedure. We address this problem and created a framework that extracts this information without the requirement of human annotations. The key idea is to track edges over several consecutive image frames and keep only the ones which are detected in multiple views. Hence, our main focus lies on Visual Odometry applications therefore, repeatability of edges in consecutive frames is a key feature and of grave importance.

In addition to that, we show that we can reduce the amount of edge information whereas we could improve processing speed, robustness and tracking accuracy on some well known datasets. Furthermore, an extensive study of several state-of-the-art edge detectors has been implemented where qualitative and quantitative evaluations have been performed. We show, that in contrast to traditional edge detector methods, machine learning based edge procedures improve robustness and accuracy significantly in most situations.
Finally our system has been evaluated on [Robust Edge-based Visual Odometry (REVO)](https://github.com/fabianschenk/REVO) real-time framework for RGB-D sensors. It has been developed at the Institute of Computer Graphics and Vision at the Technical University of Graz and is used to demonstrate the performance of the proposed method in a wide range of scenes and camera motions.

Our training framework is implemented in Python and runs on all common operating systems without any adaptions. The output can be used as training data for every machine learning based edge detector to train networks for detecting stable and repeatable edges.

## Installation
For installation I would recommend using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) package management system. It is easy to install and you can handle different environments that can have different versions of applications. This is very useful when trying out different algorithms that have different dependencies.

For the repository, use the following command:

    git clone --recurse-submodules git@github.com:huberthomas/edgetoolbox.git

### Dependencies

To install all necessary dependencies please use conda or pip. Conda is recommended because it creates its' own environment where it installs all needed packages independently from other projects.

### Conda

    conda env create -f environment.yml
    
### Python pip

    pip install -r requirements.txt --user

## Workflow

    conda activate edgetoolbox
    python reprojectEdgesFromConsecutiveFrameSet.py -h to get a list of parameters and additional information. 

Additional functions have been created through working out the thesis.

## Additional Contributions
### BDCN Stable Machine Learned Model
Fine-tuned [BDCN](https://github.com/pkuCactus/BDCN) model based on the [BSDS500](https://github.com/BIDS/BSDS500) dataset to detect the most stable edges in an image as is explained in short in the abstract. 

[Download](https://huberthomas.github.io/master/models/bdcn_sgd_30000.pth)

### Edge Detector Comparison

Web-page presenting various edge detectors on different scenes.

[Show](https://huberthomas.github.io/master)
## Datasets

[Overview](http://www.michaelfirman.co.uk/RGBDdatasets/)

||||
|---|---|---|
|[BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)|[TUM RGB-D](https://vision.in.tum.de/data/datasets/rgbd-dataset)|[NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)|
|[ETH 3D](https://www.eth3d.net/slam_overview)|[Multi-Cue](http://serre-lab.clps.brown.edu/resource/multicue/)|[Pascal Voc 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)|
|[SUN RGB-D](http://rgbd.cs.princeton.edu/)|[ICL NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)|[HDRFusion Synthetic](https://github.com/ShudaLi/HDRFusion)




















## License

See the license file.