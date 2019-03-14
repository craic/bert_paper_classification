# Using Google BERT to classify biomedical papers


### Introduction

This page and repository describes how I use Google's BERT to help classify certain biomedical scientific papers.
The data format requirements and the form of output can be confusing so this is a *detailed* walk through of my application.

It builds on a [blog post from Javed Qadrud-Din](https://blog.insightdatascience.com/using-bert-for-state-of-the-art-pre-training-for-natural-language-processing-1d87142c29e7) but goes into more of the gory details.

This describes how to set up and use BERT on a Linux machine with a GPU. My configuration is a high end linux box running 
**Ubuntu 18.04.2 LTS** with an **Nvidia GTX 1080 Ti**.

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

### Install BERT software model files

You 'install' BERT by cloning the Git repo:

```shell
git clone https://github.com/google-research/bert.git
```

There are 2 sizes of BERT Model -  **Base** and **Large** - the Base model is the only one that will fit in GPU memory.

There are 2 versions of the Base model - **Uncased** and **Cased** - choose Cased if you think the case of your text is significant.

The links for these two are:
- [BERT Base Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)
- [BERT Base Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)



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

Now you need to split your shuffled dataset into Train, Validation and Test sets. I used a typical split of 0.8 / 0.1 / 0.1 and 
just used a combination of **head** and **tail** to do this.

BERT requires **specific** names for these files - **train.tsv**, **dev.tsv** for Validation and **test.tsv**

```shell
$ wc -l all_data.tsv
17824 all_data.tsv
$ head -14260 all_data.tsv > train.tsv
$ tail -1782  all_data.tsv > test_original.tsv
$ tail -3564  all_data.tsv | head -1782 > dev.tsv
```

But hold on! The test dataset, **test.tsv**, has a slightly different format. 
It gets a header line and the only two fields included are the **ID** and **Text** - no label column. 
That's why I created **test_original.tsv** in the above step (*keep this around as it is needed later*)

[convert_tsv_to_test.py](convert_tsv_to_test.py) does this using the 'inital' test.tsv file

```shell
$ python convert_tsv_to_test.py --file test_original.tsv > test.tsv
```

That format looks like:
```text
id	text
26399369	adalimumab treatment leads to reduction of tissue ...
26004977	the effects of reference genes ...
27105521	cd30 on extracellular vesicles from ...
```

OK - we have the three input data files - train.tsv. dev.tsv and test.tsv - now we can actually use BERT


***Include a test sample of real data here ?***

#### Run BERT Fine-Tuning

There are a number of python scripts in the BERT install directory that are worth taking a look at. 
It's always worth looking at the source - but don't worry if don't understand how things are structured.
All we really need to do is run **run_classifier.py** with the suitable set of arguments.

There are a lot of arguments, but if you have done any practical machine learning before these should make sense.

This command assumes that your input files are in directory **./bert_input** and you want the output in **./bert_output**. 
It also assumes that the BERT software and model files are in the **./bert** subdirectory of your current location.

To avoid typos in a long command, I am using Javed's step of creating a shell variable with the path to the BERT model files.


```shell

$ export $BERT_BASE_DIR=./bert/uncased_L-12_H-768_A-12

$ python bert/run_classifier.py \
--task_name=cola \
--do_train=true \
--do_eval=true \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=3 \
--data_dir=./bert_input \
--output_dir=./bert_output

```

What do these arguments mean? 
- --task_name **cola** - this is the classification task
- --do_train, --do_eval - we want to train the model and evaluate it
- --vocab_file, --bert_config_file, --init_checkpoint - use these BERT model files - and in particular use this checkpoint file that represents all the wieghts from the pre-trained model that we want to fine tune
- --max_seq_length - limit the number of words in the text that we will use
- --train_batch_size - how many text records to use in each batch
- --learning_rate, --num_train_epochs - use this learning rate for this number of epochs
- --data_dir - the directory with your input data
- --output_dir - the directory where your output data will be placed 

These parameters are a good place to start. There is not much room to play with them however as you may run into out-of-memory errors.
I'll talk more about that below.

When you hit return you will see pages of output - most of which is irrelevant in most cases. 
Some of the lines show you examples of how it is tokenizing the input. Some show the individual epochs and cycles as the software runs.

You might expect there to be some --verbose / --non-verbose flag but I haven't found it.

It serves to show progress as the training proceeds.

Here is the end of a run as an example

```shell
[...]
INFO:tensorflow:Restoring parameters from ./pubmed_output_20190313/model.ckpt-2673
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2019-03-13-23:12:04
INFO:tensorflow:Saving dict for global step 2673: eval_accuracy = 0.8922559, eval_loss = 0.3347309, global_step = 2673, loss = 0.33583885
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 2673: ./pubmed_output_20190313/model.ckpt-2673
INFO:tensorflow:evaluation_loop marked as finished
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.8922559
INFO:tensorflow:  eval_loss = 0.3347309
INFO:tensorflow:  global_step = 2673
INFO:tensorflow:  loss = 0.33583885
```
This shows the script saving out a final checkpoint file from the run at global step 2673 and then reporting a final **eval_accuracy** and **eval_loss**.

The time this takes will depend on the size of your dataset, your GPU and your training parameters. On my setup this run took about 30 minutes

At this point we have an updated model based on the BERT-Base strating point that has been fine tuned with my project-specific input text.

If you look in your **output directory** that you specified you will see a bunch of files:

```shell
$ ls -l ./pubmed_output_20190313
total 5195300
-rw-rw-r-- 1 jones jones        222 Mar 13 16:11 checkpoint
drwxr-xr-x 2 jones jones       4096 Mar 13 16:12 eval
-rw-rw-r-- 1 jones jones         85 Mar 13 16:12 eval_results.txt
-rw-rw-r-- 1 jones jones    1179499 Mar 13 16:11 eval.tf_record
-rw-rw-r-- 1 jones jones   14228229 Mar 13 15:58 events.out.tfevents.1552517873.blackbox
-rw-rw-r-- 1 jones jones   14232408 Mar 13 16:11 events.out.tfevents.1552517975.blackbox
-rw-rw-r-- 1 jones jones    9476516 Mar 13 15:59 graph.pbtxt
-rw-rw-r-- 1 jones jones 1313805344 Mar 13 15:59 model.ckpt-0.data-00000-of-00001
-rw-rw-r-- 1 jones jones      22764 Mar 13 15:59 model.ckpt-0.index
-rw-rw-r-- 1 jones jones    4014894 Mar 13 15:59 model.ckpt-0.meta
-rw-rw-r-- 1 jones jones 1313805344 Mar 13 16:04 model.ckpt-1000.data-00000-of-00001
-rw-rw-r-- 1 jones jones      22764 Mar 13 16:04 model.ckpt-1000.index
-rw-rw-r-- 1 jones jones    4014894 Mar 13 16:04 model.ckpt-1000.meta
-rw-rw-r-- 1 jones jones 1313805344 Mar 13 16:08 model.ckpt-2000.data-00000-of-00001
-rw-rw-r-- 1 jones jones      22764 Mar 13 16:08 model.ckpt-2000.index
-rw-rw-r-- 1 jones jones    4014894 Mar 13 16:08 model.ckpt-2000.meta
-rw-rw-r-- 1 jones jones 1313805344 Mar 13 16:11 model.ckpt-2673.data-00000-of-00001
-rw-rw-r-- 1 jones jones      22764 Mar 13 16:11 model.ckpt-2673.index
-rw-rw-r-- 1 jones jones    4014894 Mar 13 16:11 model.ckpt-2673.meta
-rw-rw-r-- 1 jones jones    9432697 Mar 13 15:59 train.tf_record
```

I don't know what some of these represent but the **model.ckpt-2673** files represent the final checkpoint (the number on your files will be different) 
and **eval_results.txt** contains the summary accuracy and loss results.

### Varying the run parameters

[...]

### Predict the results on the Test dataset

Now we have trained our model and seen that the accuracy is as good as we can get it.

The next step is to see how it performs on the Test set which has been kept back from the software thus far.

For this you run **run_classifier.py** again but with a different set of arguments

```shell
$ export $BERT_BASE_DIR=./bert/uncased_L-12_H-768_A-12

$ export TRAINED_CLASSIFIER=./bert_output/model.ckpt-[highest checkpoint number you saw]

$ python bert/run_classifier.py \
--task_name=cola \
--do_predict=true \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=./bert_output/model.ckpt-2673 \
--max_seq_length=128 \
--data_dir=./data \
--output_dir=./bert_output/

```

What do these arguments mean? 
- --task_name **cola** - this is the classification task
- --do_predict - we want to use the model fro predictions
- --vocab_file, --bert_config_file - use these BERT model files 
- --init_checkpoint - but use the new, fine-tuned model checkpoint that we created
- --max_seq_length - this needs to be the same as you used in training
- --data_dir - the directory with your input data
- --output_dir - the directory where your output data will be placed 

This command will take your **test.tsv** file and run it through the model to produce a file called **test_results.tsv** in your output directory.


This will have, in my case, two probabilities for each line in the input file
For example

```text
0.02100023	0.97899973
0.9973598	0.002640231
0.015744895	0.9842551
0.99622524	0.0037748201
```

The first column is the probability for the **negative** state and the second for the **positive** state.

So lines 1 and 3 are predicted to be positive and lines 2 and 4 are predicted to be negative.

If you were doing a classification with multiple states I would expect this would have multiple columns.


*But how do you see if the individual predictions are correct ?*

The input **test.tsv** file did not have the original labels **AND** this output file does not include the record **ID** numbers that we are using.
This is a problem with the current version of the BERT software - but we can solve that with another custom script and the original version 
of the Test data file that we had earlier on.


[evaluate_test_set_predictions.py](evaluate_test_set_predictions.py) takes the original format test data file and the results file.

```shell
$ python evaluate_test_set_predictions.py --tsv test_original.tsv --results test_results.tsv
26004977	0	1
29767677	1	0
21240464	1	1	*
26011989	0	0	*
[...]
```

The first column is the Pubmed ID for the paper. The second column is the label that I gave it (0 = negative, 1 = positive). The third column is the
prediction and an asterisk shows lines where the prediction is correct.

I can then look up the Pubmed abstract for each paper that it got wrong and see if I can rationalize the decision. In many/most cases I can't do
that but in some I see a reveiw paper or one that describes a non-human target.




### Predict new data records

**evaluate_results.py**




