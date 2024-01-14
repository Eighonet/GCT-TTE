# GCT-TTE

![Pipeline_image](resources/TTE_pipeline_rev2_w.png#gh-light-mode-only)
![Pipeline_image](resources/TTE_pipeline_rev2_b.png#gh-dark-mode-only)

Welcome to the official repo of the GCT-TTE model -- transformer-based travel time estimation algorithm. Here we present the source code of the pipeline and demo application.

You can access the inference of our model at [gctte.online](http://gctte.online)

arXiv PDF: https://arxiv.org/abs/2306.04324 

# Prerequisites 

**Backend:** please use *application/requirements.txt* in order to compile the environment for the application. 

**Model:** the experiments were conducted with `CUDA 10.1` and `torch 1.8.1`. The following libraries must be compatible with this software setup:
```
- torch-cluster==1.6.0
- torch-geometric==2.1.0.post1
- torch-scatter==2.0.8
- torch-sparse==0.6.12
- torch-spline-conv==1.2.1
```
All other external libraries, which do not depend on `torch` and `CUDA` versions, are mentioned in `/model/requirements.txt`.

# Local tests

Launch instructions are provided [in the README file](https://github.com/Eighonet/GCT-TTE/tree/main/model) of the `/model` directory.

# Datasets

We provide two datasets corresponding to the cities of Abakan and Omsk. For each of these datasets, there are two types of target values -- real travel time (considered in this study) and real length of trip. 

<table>
<tr><th>Road network</th><th>Trips</th></tr>
<tr><td>

| | Abakan | Omsk |
|--|--|--|
|Nodes| 65524 | 231688 |
|Edges| 340012 |  1149492 |
|Clustering| 0.5278 | 0.53 |
|Usage median| 12 | 8 |
 
</td><td>

| | Abakan | Omsk |
|--|--|--|
|Trips number|  121557| 767343 |
|Coverage| 53.3% |  49.5% |
|Average time| 427 sec | 608 sec |
|Average length| 3604 m | 4216 m |

</td></tr> </table>

Provided data could be used for research purposes only. If you want to incorporate the graph data in your study, please send a request to semenova.bnl@gmail.com. The image extension can be accesed via [https://sc.link/Mw9kP](https://sc.link/Mw9kP) (Abakan) and [https://sc.link/5QWBq](https://sc.link/NL8lm) (Omsk).

# License

Established code released as open-source software under the MIT license.

# Contact us

If you have some questions about the code, you are welcome to open an issue, I will respond to that as soon as possible.

# Citation

```
﻿@Article{Mashurov2024,
        author={Mashurov, Vladimir and Chopuryan, Vaagn and Porvatov, Vadim and Ivanov, Arseny and Semenova, Natalia},
        title={GCT-TTE: graph convolutional transformer for travel time estimation},
        journal={Journal of Big Data},
        year={2024},
        month={Jan},
        day={13},
        volume={11},
        number={1},
        pages={15},
        doi={10.1186/s40537-023-00841-1},
        url={https://doi.org/10.1186/s40537-023-00841-1}
}
```
