
## RLTF: Reinforcement Learning from Unit Test Feedback <a name="corl"></a>

This is the official code for the paper **[RLTF: Reinforcement Learning from Unit Test Feedback](https://arxiv.org/pdf/2307.04349.pdf)**.
## Installation  

The code requires some dependencies as specified in `requirements.txt`. Please follow the relevant libraries to install or run: 

`pip install -r requirements.txt`


## Datasets
* **APPS**: Please follow the downloading and preprocessing instructions provided [here](https://github.com/hendrycks/apps). 
* **MBPP**: The dataset is available [here](https://github.com/google-research/google-research/tree/master/mbpp). 

Download and unzip all files into the data folder.
## Models
https://huggingface.co/Harvey6/RLTF_codet5
## Processes 

### Surprised Finetune
* **CodeT5**:  sh script/train_actor_deepspeed.sh 
* **CodeGEN**:  sh script/train_actor_codegen_deepspeed.sh

### Generating Programs Online
* **CodeT5**:  python script/generate_online_parallel.py
* **CodeGEN**:  python script/generate_codegen_online_parallel.py

### Online RL Finetune
After running the online generation for a short period and accumulating a certain number of samples：
* **CodeT5**:  sh script/train_actor_rl_online_v1_deepspeed.sh
* **CodeGEN**: sh script/train_actor_rl_codegen_online_v1_deepspeed.sh

### Generate Program, Run Unit Test, Compute pass@k
Generate Program:
* **CodeT5**:  python script/generate_parallel.py
* **CodeGEN**: python script/generate_parallel_codegen.py

Run Unit Test：
* sh script/run_unit_tests.sh

Compute pass@k：
* python compute_pass_at_k_metric.py
## Citation 

If you find the paper or the source code useful to your projects, please cite the following bibtex: 
<pre>
@misc{liu2023rltf,
      title={RLTF: Reinforcement Learning from Unit Test Feedback}, 
      author={Jiate Liu and Yiqin Zhu and Kaiwen Xiao and Qiang Fu and Xiao Han and Wei Yang and Deheng Ye},
      year={2023},
      eprint={2307.04349},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
</pre>


## License 

The code is released under BSD 3-Clause - see `LICENSE.txt` for details.

This code is developed from other open source projects: including [CodeRL](https://github.com/salesforce/CodeRL), [APPS](https://github.com/hendrycks/apps/), and [transformers](https://github.com/huggingface/transformers). We thank the original contributors of these works for open-sourcing their valuable source codes. 

