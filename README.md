# SSAN
## Introduction
This is the pytorch implementation of the **SSAN** model (see our AAAI2021 paper: [Entity Structure Within and Throughout: Modeling Mention Dependencies
for Document-Level Relation Extraction](https://arxiv.org/abs/2102.10249)).  
**SSAN (Structured Self-Attention Network)** is a novel extension of Transformer to effectively incorporate structural dependencies between input elements.
And in the scenerio of document-level relation extraction, we consider the **structure of entities**.
Specificly, we propose a transformation module, that produces attentive biases based on the structure prior so as to adaptively regularize the attention flow within and throughout the encoding stage.
We achieve SOTA results on several document-level relation extraction tasks.  
This implementation is adapted based on [huggingface transformers](https://github.com/huggingface/transformers), the key revision is how we extend the vanilla self-attention of Transformers, you can find the SSAN model details in [`./model/modeling_bert.py#L267-L280`](./model/modeling_bert.py#L267-L280).
You can also find our paddlepaddle implementation in [here](https://github.com/PaddlePaddle/Research/tree/master/KG/AAAI2021_SSAN).
<div  align="center">  
<img src="./SSAN.png" width = "466.4" height = "294.4" alt="Tagging Strategy" align=center />
</div>  


## Requirements
 * python3.6, pytorch==1.4.0, transformers==2.7.0  
 * This implementation is tested on a single 32G V100 GPU with CUDA version=10.2 and Driver version=440.33.01.


## Prepare Model and Dataset
 - Download pretrained models into `./pretrained_lm`.
For example, if you want to reproduce the results based on RoBERTa Base, you can download and keep the model files as:
```
    pretrained_lm
    └─── roberta_base
         ├── pytorch_model.bin
         ├── vocab.json
         ├── config.json
         └── merges.txt
```
Note that these files should correspond to huggingface transformers of version 2.7.0.
Or the code will automatically download from s3 into your `--cache_dir`.

 - Download [DocRED dataset](https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw) into `./data`, including `train_annotated.json`, `dev.json` and `test.json`.


## Train
 - Choose your model and config the script:  
Choose `--model_type` from `[roberta, bert]`, choose `--entity_structure` from `[none, decomp, biaffine]`.
For SciBERT, you should set `--model_type` as `bert`, and then add `do_lower_case` action.
 - Then run training script:
 
```
sh train.sh
```  
checkpoints will be saved into `./checkpoints`, and the best threshold for relation prediction will be searched on dev set and printed when evaluation.


## Predict
Set `--checkpoint` and `--predict_thresh` then run script:  
```
sh predict.sh
```
The result will be saved as `${checkpoint}/result.json`.  
You can compress and upload it to the official competition leaderboard at [CodaLab](https://competitions.codalab.org/competitions/20717#results).
```
zip result.zip result.json
```


## Citation
You can cite us as:
```
@article{Xu_Wang_Lyu_Zhu_Mao_2021, title={Entity Structure Within and Throughout: Modeling Mention Dependencies for Document-Level Relation Extraction}, volume={35}, url={https://ojs.aaai.org/index.php/AAAI/article/view/17665}, abstractNote={Entities, as the essential elements in relation extraction tasks, exhibit certain structure. In this work, we formulate such entity structure as distinctive dependencies between mention pairs. We then propose SSAN, which incorporates these structural dependencies within the standard self-attention mechanism and throughout the overall encoding stage. Specifically, we design two alternative transformation modules inside each self-attention building block to produce attentive biases so as to adaptively regularize its attention flow. Our experiments demonstrate the usefulness of the proposed entity structure and the effectiveness of SSAN. It significantly outperforms competitive baselines, achieving new state-of-the-art results on three popular document-level relation extraction datasets. We further provide ablation and visualization to show how the entity structure guides the model for better relation extraction. Our code is publicly available.}, number={16}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Xu, Benfeng and Wang, Quan and Lyu, Yajuan and Zhu, Yong and Mao, Zhendong}, year={2021}, month={May}, pages={14149-14157} }
```
