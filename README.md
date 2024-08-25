## Hi Lisa!

## Installation
`pip install -r requirements.txt`

You MUST manually obtain MMLU because it was too large to push to github >:(

```
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar xf data.tar
```

## Running
Run the `embed_sim.py` file. 
`<project_name>/embedding_similarity/implementations/full_impl/embed_sim.py`

Currently, the threshold is very low to induce matches. We also probably want to look for different/better embedding models.
