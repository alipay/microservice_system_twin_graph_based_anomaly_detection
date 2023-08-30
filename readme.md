# **Twin Graph-based Anomaly Detection via Attentive Multi-Modal Learning for Microservice System**

This is the Pytorch implementation of MSTGAD in the ASE paper: 

Requirements

* Ubuntu 18.04
* Python 3.8
* Pytorch 1.12.0
* CUDA 11.3

Dependencies can be installed by:

    pip install -r requirements.txt

## Data preparetion
The datasets (MSDS) used in this paper can be downloaded from the [links]([Multi-Source Distributed System Data for AI-powered Analytics | Zenodo](https://zenodo.org/record/3549604))

The downloaded datasets can be put in the 'data' directory.  The directory structure looks like:

    ${CODE_ROOT}
        ......
        |-- data
            |-- MSDS
                |-- concurrent_data

## Training
To preprocess the data, run:

    python util/pre_MSDS.py

To start training, run:

    python main.py





