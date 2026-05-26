# Hybrid Trigger-Invariant Pruning for Robust Backdoor Defense

!\[HTIP](fig/htip\_arch.png)

This project implements **HTIP**, a hybrid backdoor defense framework that combines trigger-aware attention-head pruning with trigger-invariant contrastive unlearning. The method first removes attention heads that are highly sensitive to trigger perturbations but less important for clean-task performance, then applies TICU-style unlearning to align clean and triggered predictions and representations.

HTIP is designed for text classification models under NLP backdoor attacks. It supports standalone TICU, standalone pruning, hybrid pruning + TICU, and comparison mode for running all defenses from the same poisoned checkpoint.

We conducted the paper experiments on a Linux server with dual Intel Xeon Gold 5220R CPUs and eight NVIDIA RTX A4000 GPUs, each with 16 GB VRAM. Below are the core dependencies used for this implementation:

```text
cuda>=12.0
python==3.10
pytorch>=2.5.1
transformers>=4.40.0
peft>=0.18.1
bitsandbytes>=0.49.1
accelerate>=1.12.0
numpy>=2.2.6
pandas>=2.3.3
scikit-learn>=1.7.2
protobuf
```

### Code Reference



### Instructions

1. **Create and Activate Conda Environment**

   Create and activate a Conda environment:

   ```bash
   conda create -n htip python=3.10 -y
   conda activate htip
   ```

2. **Install Dependencies**

   Run the following command in your terminal:

   ```bash
   pip install -r requirements.txt
   ```

   If you are using quantized LoRA models, make sure `bitsandbytes`, `accelerate`, and `peft` are installed correctly for your CUDA version.

3. **Prepare Dataset**

   Organize poisoned and clean CSV files in the following format:

   ```text
   <helper\_dir>/Data\_Poisoning/<Attack>/<Dataset>/
   ├── train\_poisoned.csv
   ├── train\_clean.csv
   ├── test\_clean.csv
   ├── test\_poisoned\_all.csv
   └── test\_poisoned\_part.csv
   ```

   Each CSV file must contain:

   ```text
   text,label
   ```

   `train\_poisoned.csv` should also contain:

   ```text
   poisoned
   ```

   If the text column is named `sentence`, `comment\_text`, or `content`, the script automatically renames it to `text`.

4. **Run HTIP with BERT-style Models**

   Example command:

   ```bash
   python3 ticu\_prune\_hybrid.py \\
     --helper\_dir ./helper \\
     --dataset SST-2 \\
     --attack HiddenKiller \\
     --model\_type auto \\
     --model\_name bert-base-uncased \\
     --defense hybrid \\
     --batch\_size 32 \\
     --epochs\_poison 5 \\
     --epochs\_unlearn 10 \\
     --acc\_threshold 0.85 \\
     --out\_dir ./prune\_ticu\_outputs
   ```

5. **Run TICU Only**

   Example command:

   ```bash
   python3 ticu\_prune\_hybrid.py \\
     --helper\_dir ./helper \\
     --dataset SST-2 \\
     --attack HiddenKiller \\
     --model\_type auto \\
     --model\_name bert-base-uncased \\
     --defense ticu \\
     --batch\_size 32 \\
     --epochs\_poison 5 \\
     --epochs\_unlearn 10 \\
     --out\_dir ./prune\_ticu\_outputs
   ```

6. **Run Trigger-Aware Pruning Only**

   Example command:

   ```bash
   python3 ticu\_prune\_hybrid.py \\
     --helper\_dir ./helper \\
     --dataset SST-2 \\
     --attack HiddenKiller \\
     --model\_type auto \\
     --model\_name bert-base-uncased \\
     --defense prune \\
     --batch\_size 32 \\
     --acc\_threshold 0.85 \\
     --prune\_step 2 \\
     --alpha\_clean 0.5 \\
     --max\_score\_batches 50 \\
     --out\_dir ./prune\_ticu\_outputs
   ```

7. **Compare TICU, Pruning, and HTIP**

   Example command:

   ```bash
   python3 ticu\_prune\_hybrid.py \\
     --helper\_dir ./helper \\
     --dataset SST-2 \\
     --attack HiddenKiller \\
     --model\_type auto \\
     --model\_name bert-base-uncased \\
     --defense compare \\
     --batch\_size 32 \\
     --epochs\_poison 5 \\
     --epochs\_unlearn 10 \\
     --out\_dir ./prune\_ticu\_outputs
   ```

   `compare` mode trains or loads one poisoned checkpoint, then runs `ticu`, `prune`, and `hybrid` from the same poisoned state.

8. **Run TICU with LoRA Models**

   Example command:

   ```bash
   CUDA\_VISIBLE\_DEVICES=0 python3 ticu\_prune\_hybrid.py \\
     --helper\_dir ./helper \\
     --dataset SST-2 \\
     --attack HiddenKiller \\
     --model\_type llama \\
     --model\_name meta-llama/Llama-2-7b-hf \\
     --defense ticu \\
     --use\_lora \\
     --load\_in\_4bit \\
     --batch\_size 4 \\
     --grad\_accum 8 \\
     --fp16 \\
     --out\_dir ./prune\_ticu\_outputs
   ```

9. **Resume from a Poisoned Checkpoint**

   ```bash
   python3 ticu\_prune\_hybrid.py \\
     --helper\_dir ./helper \\
     --dataset SST-2 \\
     --attack HiddenKiller \\
     --model\_type auto \\
     --model\_name bert-base-uncased \\
     --defense hybrid \\
     --skip\_poison\_train \\
     --resume\_path ./prune\_ticu\_outputs/auto\_SST-2\_HiddenKiller\_poisoned.pt \\
     --out\_dir ./prune\_ticu\_outputs
   ```

10. **Use Manual Triggers**

    By default, the script infers triggers from poisoned samples using `--trigger\_mode auto`. To provide triggers manually:

    ```bash
   python3 ticu\_prune\_hybrid.py \\
     --helper\_dir ./helper \\
     --dataset SST-2 \\
     --attack BadNet \\
     --model\_type auto \\
     --model\_name bert-base-uncased \\
     --defense hybrid \\
     --trigger\_mode manual \\
     --trigger "cf|mn|bb" \\
     --trigger\_multi\_sep "|" \\
     --out\_dir ./prune\_ticu\_outputs
   ```

    ### Important Arguments

    ```text
--defense              Defense mode: ticu, prune, hybrid, or compare
--acc\_threshold        Minimum validation/test clean accuracy allowed during pruning
--prune\_step           Number of candidate heads to prune per pruning trial
--alpha\_clean          Penalty for pruning clean-important heads
--max\_score\_batches    Number of batches used for head scoring; 0 uses all batches
--ticu\_gamma           Weight for prediction-invariance loss
--ticu\_delta           Weight for representation-invariance loss
--beta                 L2 regularization strength toward the initial model state
--data\_variant         Poisoned test split: poisoned\_all or poisoned\_part
--trigger\_mode         Trigger selection mode: auto or manual
--trigger              Manual trigger string; supports multiple triggers with separator
```

    ### Outputs

    The script saves poisoned and defended checkpoints in the output directory:

    ```text
<out\_dir>/<model\_type>\_<dataset>\_<attack>\_poisoned.pt
<out\_dir>/<model\_type>\_<dataset>\_<attack>\_ticu.pt
<out\_dir>/<model\_type>\_<dataset>\_<attack>\_prune.pt
<out\_dir>/<model\_type>\_<dataset>\_<attack>\_hybrid.pt
```

    It also saves pruning and result metadata:

    ```text
<out\_dir>/hybrid\_pruned\_heads.json
<out\_dir>/hybrid\_head\_ranking.json
<out\_dir>/prune\_ticu\_results.json
```

    The script prints clean accuracy and label flip rate before and after defense. For binary datasets such as SST-2 and HSOL, it reports class-wise label flip rates. For AG News, it reports the mean off-diagonal label flip rate.

    ### Notes

    If you see a CUDA device mismatch error on a multi-GPU server, run with a single GPU:

    ```bash
CUDA\_VISIBLE\_DEVICES=0 python3 ticu\_prune\_hybrid.py ...
```

    For LoRA models, make sure the following helper files are available:

    ```text
lora\_llama.py
lora\_qwen.py
lora\_mistral.py
```

    For encoder models, `--model\_type auto` with `bert-base-uncased` is recommended for reproducing the paper-style classification experiments.

    If pruning is skipped with a warning, the selected model likely does not expose a compatible `prune\_heads(...)` method. In that case, use `--defense ticu` or switch to a BERT/RoBERTa/DistilBERT-style encoder model.

    ### Citation

    If you use this code, please cite the corresponding paper:

    ```bibtex
@inproceedings{anonymous2025htip,
  title     = {Prune or Unlearn? A Hybrid Trigger-Invariant Pruning Framework for Backdoor Defense},
  author    = {Anonymous Author(s)},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  year      = {2025},
  address   = {Seoul, Korea}
}
```

