
# stocksentllm

Fine tuning an llm to predict stock sentiment based on headlines. This project attempts to train distilbert-base-uncased on [this dataset.](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) 

## Why distilbert-base-uncased?
Well, all my gpus are busy on some cuda experiments so I don't have VRAM to spare and am forced to train on my M1 Macbook. We can probably get better results on a different model. 

## Roadmap

 - [ ] Add checks for overfitting 
 - [ ] Refactor, isolate and clean up training and inference code 
 - [ ] Preferably find a better model
 - [ ] Switch from training a full model to a LoRA or QLoRA
 - [ ] More training data
 - [ ] Add way to pull in headlines from GNews (dreading using bs4)

Not super well versed in the llm game (*yet*), so if anyone wants to help or has ideas feel free.
