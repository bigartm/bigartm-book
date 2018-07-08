#! /bin/bash

bigartm -c vw.txt -v vocab.txt --cooc-window 10 --cooc-min-tf 5 --write-cooc-tf cooc_tf_ --cooc-min-df 5 --write-cooc-df cooc_df_ --write-ppmi-tf ppmi_tf_ --write-ppmi-df ppmi_df_
