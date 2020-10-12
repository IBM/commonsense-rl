# TextWorld Commonsense (TWC)

TextWorld Commonsense (TWC) is a new text-based environment for RL agents that requires the use of commonsense knowledge from external knowledge sources to solve challenging problems. This repository provides the code for the work described below.

## Text-based Reinforcement Learning Agents with Commonsense Knowledge: New Challenges, Environments and Baselines
---------------------------------------------------------------------------
TextWorld Commonsense (TWC) dataset/environment and code for the sample RL agents reported in the paper 
[Text-based RL Agents with Commonsense Knowledge: New Challenges, Environments and Baselines](https://arxiv.org/abs/2010.03790).

![TWC](./games/twc/twc_cleanup.png?raw=true "TextWorld Commonsense")


## Prepare TWC Environment

Install the following requirements:
```
# Dependencies
conda create -n twc python=3.7 numpy scipy ipython matplotlib
conda activate twc
conda install pytorch=1.3.0 torchvision cudatoolkit=9.2 -c pytorch
pip install textworld==1.2.0
conda install  nltk gensim networkx unidecode
pip install -U spacy
python -m spacy download en_core_web_sm
python -m nltk.downloader 'punkt'
```

Create the following directories in the main folder for the storing the model, result and log files.
```
mkdir results
mkdir logs
```

Download the Numberbatch embedding for the knowledge-aware agents:
```
cd embeddings
Download https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
gzip -d numberbatch-en-19.08.txt.gz
```
## IQA Cleanup Games

This directory contains a set of 45 games created based on the dataset available in the folder ```iqa_dataset``` (under ```game_generation/iqa_cleanup``` directory). 
The games are grouped into three difficulty levels as follows.

* **Easy level**: games in the ```easy``` directory have only *1 room and 1 object* that need to be placed in
the appropriate location.
* **Medium level**: games with a ```medium``` difficulty level have _1_ room and _3_ objects shuffled across the room. Given the high number of objects,
    in this case we expect that the agent will need to rely more on the commonsense knowledge graph.
* **Hard level**: games with a ```hard``` difficulty level have _1_ or _2_ rooms with 6 or 7 objects shuffled across the rooms. In this case, we expect that the agent needs to leverage other knowedge sources such as historical state description etc efficiently in addition to the commonsense knowledge.
    

#### Custom Game Generation
You can use the benchmark games provided in the ```games/twc``` folder for your custom agents. 
You may use the ```game_generation``` folder to generate customized games generated with 
commonsense knowledge for your agents. See the [instruction](game_generation/README.md) for more details.



## Commonsense Knowledge Aware Agents


**To run Text-only agent**, use the following command:

<code>
python -u train_agent.py --agent_type simple --game_dir ./games/twc --game_name *.ulx --difficulty_level easy
</code>

The above command uses state description/observation  (with GloVe embedding for the word representation by default)
to select the next action.


**To run Text + Commonsense agent**, use the following command:


<code>
python -u train_agent.py --agent_type knowledgeaware --game_dir ./games/twc --game_name *.ulx --difficulty_level easy --graph_type world --graph_mode evolve --graph_emb_type numberbatch --world_evolve_type CDC
</code>

The above command uses state description/observation  (with GloVe embedding for the word representation by default) and
the commonsense subgraph extracted from ConceptNet using Contextual Direct Connection (CDC) method 
(with Numberbatch embedding for the graph representation)
to select the next action.

Use `test_agent.py` with `--split` and `--pretrained_model` options to test the above agents.

## Sample Results

We give some sample results for `Text` and `Text + Commonsense` RL agents against the `Human` and the `Optimal`
performances. The `Text`-based RL agent uses only the state description/observation only for selecting the next
action, where as, `Text + Commonsense`-based RL agent uses both the state description and commonsense knowledge graph
to select next action (See our paper for more details).


#### Difficulty level: Easy
| Agents  | #Steps | Normalized Score |
| ------------- | ------------- | ------------- |
| `Text`  | 17.59 ± 3.11  | 0.86 ± 0.04  |
| `Text + Commonsense` | 14.43 ± 3.08  | 0.93 ± 0.06  |
| `Human`  | 2.12 ± 0.49  | 1.0 ± 0.00  |
| `Optimal`  | 2.0 ± 0.00  | 1.0 ± 0.00  |

#### Difficulty level: Medium
| Agents  | #Steps | Normalized Score |
| ------------- | ------------- | ------------- |
| `Text`  | 37.99 ± 6.03  | 0.74 ± 0.11  |
| `Text + Commonsense` | 25.11 ± 2.33  | 0.87 ± 0.04  |
| `Human`  | 5.33 ± 2.06  | 1.0 ± 0.00  |
| `Optimal`  | 3.60 ± 0.55  | 1.0 ± 0.00  |

#### Difficulty level: Hard
| Agents  | #Steps | Normalized Score |
| ------------- | ------------- | ------------- |
| `Text`  | 49.21 ± 0.58  | 0.54 ± 0.04  |
| `Text + Commonsense` | 43.27 ± 0.70  | 0.45 ± 0.00  |
| `Human`  | 15.00 ± 3.29  | 1.0 ± 0.00  |
| `Optimal`  | 15.00 ± 2.00  | 1.0 ± 0.00  |

The above results confirm that there is still much progress to be made in
retrieving and encoding the commonsense knowledge effectively to solve `Text + Commonsense` problems; 
and we hope that TWC can spur further research in this direction for the text-based RL.

## Bibliographic Citations
If you use our TWC environment and/or the code, please cite us:
```
@article{murugesan2020textbased,
      title={Text-based RL Agents with Commonsense Knowledge: New Challenges, Environments and Baselines}, 
      author={Keerthiram Murugesan, Mattia Atzeni, Pavan Kapanipathi, Pushkar Shukla, Sadhana Kumaravel, Gerald Tesauro, Kartik Talamadupula, Mrinmaya Sachan and Murray Campbell},
      year={2020},
      eprint={2010.03790},
      archivePrefix={arXiv},
      journal={CoRR},
      volume={abs/2010.03790}
}
```
Please share _commonsense-rl_ using http://ibm.biz/commonsense-rl
#### Feedback

* [File an issue](https://github.com/IBM/commonsense-rl/issues/new) on GitHub.
* Ask a question on [Stack Overflow](https://stackoverflow.com/questions/tagged/commonsense-rl%20twc?sort=Newest&edited=true).

#### Relevant Resources 

* ConceptNet http://conceptnet.io/
* Microsoft TextWorld https://www.microsoft.com/en-us/research/project/textworld/
  
