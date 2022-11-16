# GCT-TTE

![Pipeline_image](resources/transtte_pipeline_wh.png#gh-light-mode-only)
![Pipeline_image](resources/transtte_pipeline_bl.png#gh-dark-mode-only)

Welcome to the official repo of the GCT-TTE model -- transformer-based travel time estimation algorithm. Here we present the source code of the pipeline and dedicated application.

Vaagn Chopuran, Vladimir Mashurov, [Vadim Porvatov](https://www.researchgate.net/profile/Vadim-Porvatov), Arseny Ivanov, Ksenia Kuznetsova, [Natalia Semenova](https://www.researchgate.net/profile/Natalia-Semenova-7)

You can access inference of our model at [mtte.online](http://mtte.online)

arXiv PDF: to be added

# Prerequisites

**Backend:**

```

```

**Model:**

```

```

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
|Trips number|  119986 | 120000 |
|Coverage| 0.535 |  0.392 |
|Average time| 433.61 | 622.67 |
|Average length| 3656.34 | 4268.72 |

</td></tr> </table>

Provided data could be used for research purposes only. If you want to incorporate it in your study, please send request to semenova.bnl@gmail.com.

# License

Established code released as open-source software under the MIT license.

# Contact us

If you have some questions about the code, you are welcome to open an issue, I will respond to that as soon as possible.

# Citation

To be updated.

```
```
