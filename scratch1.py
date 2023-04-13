"""
extract the tweets and the language. for each tweet check language (from the col) and pass to
either roberta or robertuito.

can use xlm-roberta and put the sexism label at the end of the sentence. in inference we can put 
a mask at the end of the sentence and let it predict and extract that label.

shuffle the training set.

might have to train the xlm-roberta model on the sexism dataset. for regular roberta/robertuito 
fine-tuning should be enough.
"""
