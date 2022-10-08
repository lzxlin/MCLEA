# MCLEA

The code and dataset of paper _**Multi-modal Contrastive Representation Learning for Entity Alignment**_ [[arxiv](https://arxiv.org/abs/2209.00891)] [[acl]()] in Proceedings of COLING 2022 (oral).

## Dataset

### Bilingual datasets

The multi-modal version of DBP15K dataset comes from the [EVA](https://github.com/cambridgeltl/eva) repository, and the folder `pkls` of DBP15K image features should be downloaded according to the guidance of EVA repository, and the downloaded folder `pkls` is placed in the `data` directory of this repository.

The word embedding we used is `glove-6B`, you can download it from [glove](https://nlp.stanford.edu/data/glove.6B.zip), and unzip it into the `data/embedding` directory.

### Cross-KG datasets

The original cross-KG datasets (FB15K-DB15K/YAGO15K) comes from [MMKB](https://github.com/mniepert/mmkb), in which the image embeddings are extracted from the pre-trained VGG16. We use the image embeddings provided by [MMKB](https://github.com/mniepert/mmkb#visual-data-for-fb15k-yago15k-and-dbpedia15k) and transform the data into the format consistent with DBP15K. The converted dataset can be downloaded from [BaiduDisk](https://pan.baidu.com/s/1MLGBNyFjb9LLa4urCk4hCA) (the password is `stdt`), and placed them in the `data` directory.

## Training MCLEA

### Bilingual datasets

Here is the example of training MLCEA on `DBP15K`.

```bash
bash run_dbp15k.sh 0 42 zh_en
bash run_dbp15k.sh 0 42 ja_en
bash run_dbp15k.sh 0 42 fr_en
```

### Cross-KG datasets

Here is the example of training MCLEA on `FB15K_DB15K` with different ratio seeds. Similarly, you can replace the parameter `FB15K_DB15K` with `FB15K_YAGO15K` to train FB15K-YAGO15K dataset.

```bash
bash run_mmkb.sh 0 42 FB15K_DB15K 0.2
bash run_mmkb.sh 0 42 FB15K_DB15K 0.5
bash run_mmkb.sh 0 42 FB15K_DB15K 0.8
```

## Citation

If you use this model or code, please cite it as follows:

```bash
@inproceedings{lin2022multi, 
  title = {Multi-modal Contrastive Representation Learning for Entity Alignment},
  author = {Lin, Zhenxi and Zhang, Ziheng and Wang, Meng and Shi, Yinghui and Wu, Xian and Zheng, Yefeng}, 
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
  year = {2022},
}
```

## Acknowledgement

Our codes are modified based on [EVA](https://github.com/cambridgeltl/eva), and we would like to thank their open-sourced work.
