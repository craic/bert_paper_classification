# Using Google BERT to classify biomedical papers

### Background

BERT (Bidirectional Encoder Representations from Transformers) is a language representation model from Google that can be applied to
a number of Natural Language Processing (NLP) tasks with considerable success.

The details are described in the [Original BERT paper](https://arxiv.org/abs/1810.04805) and in the code contained in the 
[BERT Github repository](https://github.com/google-research/bert).

I was interested in using BERT to help identify scientific papers in the [Pubmed](https://www.ncbi.nlm.nih.gov/pubmed/) database that described
work on human therapeutic antibodies.

I operate a commercial database on therapeutic antibodies - [TABS](https://tabs.craic.com) - for biotechnology companies working in the field.
As part of that work, I use keywords and term frequencies to identify relevant new papers in downloads from Pubmed. These simple techniques help
highlight potential papers when I manually screen the sets of titles and abstracts.

My initial attempts to use machine learning (LSTM, CNN, etc) to help classify papers were not encouraging. Aside from the specific algorithms, there
are issues inherent to the data sources that make this a difficult task:

- Abstracts are often short and may not summarize the paper well.
- Some papers describe antibodies used in diagnostic rather than therapeutic applications. 
- Some papers are review articles rather than contain original research. 
- Some papers refer to therapeutic antibodies against infectious agents, rather than human antigens. 

These papers are not of interest - and yet they share many of the same keywords with relevant papers.

BERT, and related projects like OpenAI GPT and ELMo, are attractive as they pre-train their models on very large amounts of text, for example all of Wikipedia.

They have the potential for a form of Transfer Learning using custom datasets such as my antibody datasets.

Transfer Learning is widely used in Image Classification and, in that domain, involves initial training on very large sets of images representing a 
diverse set of objects, which is computationally very expensive. The resulting models can then be *fine-tuned* using custom datasets to produce 
models that are very effective at classifying the subset of objects. This fine-tuning step is relatively cheap and quick to perform.

BERT allows the same general approach to be used in NLP projects - and this is what attracted my attention.

The BERT software is necessarily complex and, for my level of expertise, not at all easy to approach.

However, I stumbled across a 
[blog post from Javed Qadrud-Din](https://blog.insightdatascience.com/using-bert-for-state-of-the-art-pre-training-for-natural-language-processing-1d87142c29e7)
 that describes how to use BERT for document classification in an application that is similar to my own.

Without this post, I would have struggled to get my application off the ground - so I am *very* grateful to Javed!

Even so, I still had to work out a lot of details before I could get this approach to work with my data.
So this page and this repository is my contribution to making this process easier and helping others to use it.


I am not going to describe how BERT works - or how to install it here - please see 
[Javed's post](https://blog.insightdatascience.com/using-bert-for-state-of-the-art-pre-training-for-natural-language-processing-1d87142c29e7) for that.


### Data Preparation

The structure of BERT input and output files and the specific commands that you run are, well, *not very elegant*... Hopefully this will change
as the software matures, but for now you have to deal with some quirks in order to get data in and out.


#### Input data

My datasets start their life as XML files but internally I convert these into YAML files where each record has a unique **ID**, which refers back to 
Pubmed, and **Text**, which is a composite of the paper Title, Abstract and Authors. For example:

```yaml
- id: '28121495'
  text: Belimumab for the treatment of recalcitrant cutaneous lupus. Belimumab cutaneous
    lupus refractory Background Belimumab is a monoclonal antibody that reduces B
    lymphocyte survival by blocking the binding of soluble human B lymphocyte stimulator
    (BLyS) to its B cell receptors. The utility of belimumab for management of resistant
    [...]
 ```

I have one file of **Positive** documents, which describe therapeutic antibodies, and **Negative** documents, which describe any others 
that refer to antibodies.


As with most machine learning models, BERT requires 3 input files - Training Data, Validation Data and Test Data. 
These need to be in **TAB delimited format** with one record per line.

Training and Validation data have one format *...but Test data has a slightly different format...*

My approach was to:

- Take all my input data and convert it to the Training/Validation format
- Split that into sets for Training/Validation/Test using a ratio of 0.8 / 0.1 / 0.1
- Convert the Test set into that specific format

The scripts that I use for this are provided in this repository.

#### Convert YAML to Training data

The Training Data format has 4 columns, separated by a **Tab** (\t) character
- column 1 - a unique ID
- column 2 - an integer label - my dataset uses 0 for a negative paper and 1 for a positive paper
- column 3 - a dummy column where each line has the same letter (in this case 'a') - perhaps this is used in other NLP tasks
- column 4 - the text, which has had tabs and newlines stripped out of it.

Here is an example (with the text truncated) where the first 2 lines are negative and the last 2 are positive

```text
27431639	0	a	[a case of portal vein thrombosis occurring during capeox ...
24117520	0	a	corticosterone targets distinct steps of synaptic transmission ...
27768015	1	a	hemolytic uremic syndrome in children.  hemolytic uremic syndrome ...
28162025	1	a	secukinumab in the treatment of psoriasis: an update.  ...
```

The script that does this conversion is [convert_yml_to_tsv.py](convert_yml_to_tsv.py).
I run this on both input files, concatenate them and then shuffle all the rows using the **shuf** command, which is found in the **gnu coreutils** 
and which should be available on most Linux systems.



Given separate **positive** and **negative** files, this would be used like this:
```shell
$ python convert_yml_to_tsv.py --file negative.yml --label 0 >  tmp.tsv
$ python convert_yml_to_tsv.py --file positive.yml --label 1 >> tmp.tsv
$ shuf tmp.tsv > all_data.tsv
```




#### Overview of the BERT software


