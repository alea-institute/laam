
# LaaM - Linux as a Model
What happens when we train a simple transformer model to memorize the GPL2 source of the Linux kernel?
 
## Motivation
Simply put, the OSI is making a grave mistake by ignoring the most important transitive dependency in AI - the training data.

As of the latest version of [The Open Source AI Definition (draft v. 0.0.8)](https://opensource.org/deepdive/drafts/the-open-source-ai-definition-draft-v-0-0-8), 
the OSI has decided that the legal status of training data is irrelevant to their subsequent "approval" of models as "open."

The argument in favor of this omission is that such a requirement would be inconvenient and legally ambiguous
in some jurisdictions. 

This would be like Creative Commons encouraging the authors of textual or audiovisual works to ignore
the terms of copyleft licenses in their work.

**Simply put, organizations like the OSI must take a clear, common sense stance: "AI" models like text or multimodal LLMs
cannot be considered "open" if they are trained on "stolen" or "closed source" data.**    

## Details
To demonstrate how ridiculous the OSI's position is, I have trained simple transformer models to memorize the
source code of Linux version 1.0, which is licensed under the GPL2.

This model is documented and trained in perfect compliance with the OSI's draft guidance on Data Information, Code, 
and Model sections.  All source code is available in this repository, all dependencies are open source,
all input training data is directly described by the source code, and all model weights are available on 
Hugging Face.

## Example Model - 5M parameter Llama2 architecture
For example, this 5M parameter model can be trained on practically any device in a few hours.  The model trivially 
emits copies of Linux 1.0 source code.  For example, using the HuggingFace hub copy at `mjbommar/linux-as-a-model-5M`:

```python
>>> from transformers import pipeline
>>> p = pipeline('text-generation', 'mjbommar/linux-as-a-model-5M')
>>> print(p('', max_new_tokens=256, do_sample=True, temperature=0.2)[0]['generated_text'])
 linux/drivers/net/3c503.c /* 3c503.c: A shared-memory NS8390 ethernet driver for linux. */
/*
 Written 1992,1993 by Donald Becker.

 Copyright 1993 United States Government as represented by the
 Director, National Security Agency. This software may be used and
 distributed according to the terms of the GNU Public License,
 incorporated herein by reference.

 This driver should work with the 3c503 and 3c503/16. It should be used
 in shared memory mode for best performance, although it may also work
 in programmed-I/O mode.

 The Author may be reached as becker@super.org or
 C/O Supercomputing Research Ctr., 17100 Science Dr., Bowie MD 20715
*/

```

## License
For the sake of demonstration, I have licensed the model source **and weights** under the MIT terms,
and the OSI should support this model as completely open and compliant with their draft guidance.


## Train your own model
```
# ensure poetry available
# curl -sSL https://install.python-poetry.org | python3 -

# setup poetry environment
$ poetry install --no-root

# optionally install flash-attn
# poetry run pip install wheel
# MAX_JOBS=4 poetry run pip install flash-attn --no-build-isolation

# train a tokenizer with fixed vocab size on linux version 1.0
$ PYTHONPATH=. poetry run python3 -m laam.commands.train_tokenizer \
    --version v1.0/1.0 \
    --vocab-size 32768

# train a 5M parameter model on it

# stage 1: large batch size, 1e-3 learning rate to safely converge near solution 
$ PYTHONPATH=. poetry run accelerate launch \
    laam/commands/train_llama.py \
    --version v1.0/1.0 \
    --precision bfloat16 \
    --hidden_size 64 \
    --intermediate_size 256 \
    --num_hidden_layers 8 \
    --num_attention_heads 32 \
    --max_position_embeddings 512 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --epochs 100
    
# stage 2: single sample batches with 1e-4 learning rate to memorize
$ PYTHONPATH=. poetry run accelerate launch \
    laam/commands/train_llama.py \
    --version v1.0/1.0 \
    --precision bfloat16 \
    --reload \
    --learning_rate 0.0001 \
    --batch_size 1 \
    --epochs 100
``` 