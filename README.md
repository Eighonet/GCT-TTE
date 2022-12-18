# GCT-TTE

![Pipeline_image](resources/TTE_pipeline_rev2_w.png#gh-light-mode-only)

Welcome to the official repo of the GCT-TTE model -- transformer-based travel time estimation algorithm. Here we present the source code of the pipeline and dedicated application.

*anonymized placeholder for the list of authors*

You can access the inference of our model at [gctte.online](http://gctte.online)

arXiv PDF: to be added

# Prerequisites

**Backend:** please use *application/requirements.txt* in order to compile the environment for the application. 

**Model:**

Experiments were conducted with `CUDA` 10.1 and `torch 1.8.1`. The following libraries must be compatible with this software setup:

- torch-cluster==1.6.0
- torch-geometric==2.1.0.post1
- torch-scatter==2.0.8
- torch-sparse==0.6.12
- torch-spline-conv==1.2.1

All other external libraries, which do not depend on `torch` and `CUDA` versions, are mentioned in `/model/requirements.txt`.

# Local test


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

Provided data could be used for research purposes only. If you want to incorporate it in your study, please send request to *anonymized placeholder*.

# License

Established code released as open-source software under the MIT license.

# Contact us

If you have some questions about the code, you are welcome to open an issue, I will respond to that as soon as possible.

# Citation

To be updated.

```
```
