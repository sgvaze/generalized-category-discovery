# Generalized Category Discovery

This repo contains code for our paper: [Generalized Category Discovery](https://www.robots.ox.ac.uk/~vgg/research/gcd/)

Given a dataset, some of which is labelled, *Generalized Category Discovery* is the task
of assigning a category to all the unlabelled instances. Unlabelled instances could come from labelled or 'New' classes.

![image](https://github.com/sgvaze/generalized-category-discovery/blob/main/assets/main_img.png)

## Contents
[:boom: 1. Updates](#updates)

[:running: 2. Running](#running)

[:1234: 3. Results](#results)

[:clipboard: 4. Citation](#cite)

## <a name="updates"/> :boom: Updates

### Updates to paper since pre-print (updated PDF available [here](https://www.robots.ox.ac.uk/~vgg/research/gcd/resources/generalized_category_discovery.pdf), ArXiv updating soon)

* We introduced a more rigorous evaluation metric - when computing ACC, we compute the Hungarian algorithm only once across all unlabelled data.
   * This single set of linear assignments is then used to compute ACC on 'Old' and 'New' class subsets (see Appendix E)
   * Practically, this involves switching from 'v1' to 'v2' evaluation in ```./project_utils/cluster_and_log_utils.py```

## <a name="running"/> :running: Running

### Dependencies

```
pip install -r requirements.txt
```

### Config

Set paths to datasets, pre-trained models and desired log directories in ```config.py```

Set ```SAVE_DIR``` (logfile destination) and ```PYTHON``` (path to python interpreter) in ```bash_scripts``` scripts.

### Datasets

We use fine-grained benchmarks in this paper, including:                                                                                                                    
                                                                                                                                                                  
* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

We also use generic object recognition datasets, including:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet](https://image-net.org/download.php)


### Scripts

**Train representation**:

```
bash bash_scripts/contrastive_train.sh
```

**Extract features**: Extract features to prepare for semi-supervised k-means. 
It will require changing the path for the model with which to extract features in ```warmup_model_dir```

```
bash bash_scripts/extract_features.sh
```

**Fit semi-supervised k-means**:

```
bash bash_scripts/k_means.sh
```

### Note on semi-supervised k-means
Under the old evaluation metric ('v1') we found that semi-supervised k-means consistently boosted performance
over standard k-means, on 'Old' and 'New' data subsets. 
When we changed to 'v2' evaluation, we re-evaluated models in Tables {2,3,5} 
(including the ablation) and updated the figures.

However, recently, we have found that SS-k-means can be sensitive to bad initialisation under 'v2', and can 
sometimes *lower* performance on some datasets. Increasing the number of inits for SS-k-means can help. 
We are investigating this further now - suggestions and PRs welcome!

## <a name="results"/> :1234: Results

Results from re-running models with this repo compared to reported numbers:

| **Dataset**       | **All** | **Old** | **New** |
|---------------|------------|---------------|-----------|
| Stanford Cars (paper) | 39.0 | 57.6 | 29.9 |
| Stanford Cars (repo) | 39.9 | 58.5 | 30.9 |
| CIFAR100 (paper) | 70.8 | 77.6 | 57.0 |
| CIFAR100 (repo) | 71.3 | 77.4 | 59.1 |

## <a name="cite"/> :clipboard: Citation

If you use this code in your research, please consider citing our paper:
```
@InProceedings{vaze2022gcd,
               title={Generalized Category Discovery},
               author={Sagar Vaze and Kai Han and Andrea Vedaldi and Andrew Zisserman},
               booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
               year={2022}}
```
