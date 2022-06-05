# NER-Systems
Implementation of NER systems:

## 1. BiLSTM NER 
File ``ner_bisltm.py`` includes the implementation of a NER system using Keras and a Bidirectional LSTM neural network.

## 2. Transformer-based NER
File ``finetune_bioBert_CRAFT+Gazetter.ipynb`` includes the training of a transformers-based NER model with a custome dataset for the **Biomedical domain**. It uses the original **CRAFT** dataset (not included in the HuggingFace datasets, and an additional gazetteer, which is a list of entities manually created). The [CRAFT](https://github.com/UCDenver-ccp/CRAFT) dataset's cheme is replaced from three word letter to full entity name (i.e., TAX to Taxon). The gaazetteer is not provided in this repository due to confidentiality issues. The custom dataset is concatenated and a HuggingFace dataset is created:

  ``` 
  from datasets import Dataset
  dataset_val = Dataset.from_pandas(df_val)
  dataset_train=Dataset.from_pandas(df_train)
  dataset_test=Dataset.from_pandas(df_test)
  ```
  
The dataset's tags are: 
```['I-Protein', 'I-Cell', 'I-Chemical', 'B-GENE', 'B-Cell', 'B-Chemical', 'I-Sequence', 'I-Taxon', 'O', 'B-Taxon', 'B-Protein', 'B-Sequence', 'I-GENE']```

It uses **BioBERT** as the transformer architecture.
