# X-TAIL (Cross-domain Task-Agnostic Incremental Learning)

Eleven-dataset continual benchmark:  
**Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397**


## Data

Prepare datasets per [ZSCL `datasets.md`](https://github.com/Thunderbeee/ZSCL/blob/main/mtil/datasets.md) and place them under `data/` (or set `XTAIL_DATA_ROOT`).

## Training

Edit paths in `run_train.sh`, then:

```bash
bash run_train.sh
```

## Evaluation

```bash
bash run_eval.sh
```

Summarize the accuracy matrix: [X-TAIL](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning)

```bash
python3 -m src.xtail_accuracy --log-path output_eval_clean.txt --num-tasks 11
```

