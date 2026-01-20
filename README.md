# PATCH: Mitigating PII Leakage in Language Models with Privacy-Aware Targeted Circuit PatcHing



<!-- add a picture -->
![PATCH](
   main.png
)

## Fine-Tuning

We show how to fine-tune a [Huggingface](https://huggingface.co/gpt2) model on the [ECHR](https://huggingface.co/datasets/ecthr_cases) dataset (i) without defenses, and (ii) with differentially private training.

**Build & Run**

We recommend setting up a conda environment for this project.
```shell
$ conda create -n pii-leakage python=3.10
$ conda activate pii-leakage
$ pip install -e .
```

To install fastDP,

```shell
cd libs/fast-differential-privacy
python -m setup develop
```

**Run fine-tuning**

Add a wandb key to export WANDB_API_KEY= in ./finetune.sh.
We have all the commands in finetune.sh. 
We can run the following:

```shell
./finetune.sh
```

## Mechanistic Interpretability Analysis

To install EAP-IG,

```shell
conda create -n py312 python=3.12
conda activate py312
cd libs
git clone https://github.com/hannamw/EAP-IG.git
cd EAP-IG
pip install .
pip install cmapy
pip install seaborn==0.13.2
```

**Analyzing Impact of DP on General Circuits**

You can run the shell script with all the commands to generate relevant csv files
```shell
./gencircuits.sh
```

## Attack

Assuming your fine-tuned model is located at ```../echr_undefended``` run the following attacks.
Otherwise, you can edit the ```model_ckpt``` attribute in the ```../configs/<ATTACK>/echr-gpt2-small-undefended.yml``` file to point to the location of the model.

**PII Extraction**

This will extract PII from the model's generated text.
```shell
$ python extract_pii.py --config_path ../configs/pii-extraction-echr-qwen3-17-baseline-loc.yml
```

## Credits

- Use [modified code](https://github.com/jyhong836/pii_leakage) of [pii-leakage code](https://github.com/microsoft/analysing_pii_leakage)
- [EAP-IG](https://github.com/hannamw/eap-ig-faithfulness/tree/main)
