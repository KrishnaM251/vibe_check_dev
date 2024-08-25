## Hi Lisa!

## Installation
`pip install -r requirements.txt`

You MUST manually obtain MMLU because it was too large to push to github >:(

```
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar xf data.tar
```

## Running
- Make sure that the "clean_battle_conv..." json file is in renamed to `conv_battle.json` the following folder:  
  
`embedding-simililarity/clean_battle/clean_battle_data`  

- Also make sure that the "mmlu dataset..." folder is in renamed to `mmlu_data`. Please replace the empty folder at the following path:  

`embedding-similarity/data/mmlu_data/`  


- Run the `embed_sim.py` file. 

- Currently, the threshold is very low to induce matches. We also probably want to look for different/better embedding models.
