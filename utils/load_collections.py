from sklearn.datasets import fetch_20newsgroups
import os
import re

def load_20newsgroups(path='data'):
    """
    Download train part of 20newsgroups collection. Simple preprocess it, 
    convert to vowpal wabbit format, save in path/20newsgroups directory.
    
    This function does nothing if file path/20newsgroups/20newsgroups_train.vw
    already exists.
    
    Parameters:
    -----------
    path: str
        The folder for the collection saving
    """
    if os.path.isfile('{}/20newsgroups/20newsgroups_train.vw'.format(path)):
        return None
    data = fetch_20newsgroups(data_home=path, subset='train', remove=('headers', 'footers', 'quotes'))
    if os.path.isfile('{}/20newsgroups'.format(path)):
        os.remove('{}/20newsgroups'.format(path))
    if not os.path.isdir('{}/20newsgroups'.format(path)):
        os.mkdir('{}/20newsgroups'.format(path))
    with open('{}/20newsgroups/20newsgroups_train.vw'.format(path), 'w') as f_output:
        for i, (document, document_class) in enumerate(zip(data['data'], data['target'])):
            content = " ".join(re.sub('[^a-z]', ' ', document.lower()).split())
            new_line = '{} |text {} |class_id {}\n'.format(i, content, document_class)
            f_output.write(new_line)
    os.remove('{}/20news-bydate.pkz'.format(path))