<h1 align="center"><b>PyGNN</b></h1>
<p align="center">
    <a href="https://dl.acm.org/doi/10.1145/3580305.3599478"><img alt="Publication" src="https://img.shields.io/static/v1?label=Pub&message=KDD%2723&color=purple"></a>
    <a href="https://github.com/hygeng/PyGNN/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-yellow" alt="PRs"></a>
    <a href="https://github.com/hygeng/PyGNN/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/hygeng/PyGNN?color=green"></a>
    <a href="https://github.com/hygeng/PyGNN/stargazers"><img src="https://img.shields.io/github/stars/hygeng/PyGNN?color=red&label=Star" alt="Stars"></a>
</p>

Official Implementation of KDD 2023 paper:    
"Pyramid Graph Neural Network: A Graph Sampling and Filtering Approach for Multi-scale Disentangled Representations"

## PyGNN

We propose Pyramid Graph Neural Network framework(PyGNN), a multi-scale framework for node representations on graphs.     
More details are shown in this [project](https://hygeng.top/publication/2023kdd-pygnn/).
![](resources/fig/PyGNN-Framework.png)

### Environment Settings    
```
* pytorch               1.7.0    
* numpy                 1.18.1    
* torch-geometric       1.7.0    
* torch-cluster         1.6.0    
* torch-scatter         2.0.5    
* torch-sparse          0.6.8    
* torch-spline-conv     1.2.0    
* scipy                 1.6.2    
```

(The currently used dataset may not support later torch-geometric versions.)    

### Run
1. set "TOP_DIR" as dataset directory in src/dataloader.py (provided in "./dataset/")    
2. Generate Pyramid subgraphs (provided)    
```
python src/proc/Downsample.py -d ${dataname}
python src/proc/post_proc.py -d ${dataname} -s Downsample
```
3. run:
```
bash node.sh
```


## Citation

If you use our implementation in your works, we would appreciate citations to the paper:

```bibtex
@inproceedings{geng2023pyramid,
    author = {Geng, Haoyu and Chen, Chao and He, Yixuan and Zeng, Gang and Han, Zhaobing and Chai, Hua and Yan, Junchi},
    title = {Pyramid Graph Neural Network: A Graph Sampling and Filtering Approach for Multi-Scale Disentangled Representations},
    year = {2023},
    isbn = {9798400701030},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3580305.3599478},
    doi = {10.1145/3580305.3599478},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    pages = {518–530},
    numpages = {13},
    keywords = {graph algorithms, spectral graph theory, graph neural networks},
    location = {Long Beach, CA, USA},
    series = {KDD '23}
}
```


