from utils import *
import os
import multiprocessing

#save model
model_dir = 'baseline_best_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

train_path=r'C:\Users\11247\Desktop\coursework\nlp\Meta-GPT Style Transfer\Baseline\Dataset_1\Meta_training\train_combined.tsv'
val_path=r'C:\Users\11247\Desktop\coursework\nlp\Meta-GPT Style Transfer\Baseline\Dataset_1\Meta_training\val_combined.tsv'

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_n_val(train_path=train_path,
                val_path=val_path,
                optimizer_key='Adam',
                model_key='GPT-2',
                tokenizer_key="GPT",
                batch_size=40,
                num_epoch=50,
                patience=5,
                model_dir=model_dir)