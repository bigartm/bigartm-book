# -*- coding: utf-8 -*-

import shutil

def extract_vw_subsample(in_vw_filename, out_vw_filename, num_docs=10000):
	if num_docs == -1:
		shutil.copyfile(in_vw_filename, out_vw_filename)

	with open(in_vw_filename, 'r') as fin:
		with open(out_vw_filename, 'w') as fout:
			for i, line in enumerate(fin):
				fout.write(line)

				i += 1
				if i == num_docs:
					break


def bigartm_vw2matrix_market(vw_filename, mm_data_filename, mm_vocab_filename):
    D = 0
    NNZ = 0
    token2id = {}
    id2token = {}
    id2counter = {}

    with open(vw_filename, 'r') as fin:
        index = 1
        for line in fin:
            D += 1
            for elem in line.strip().split(' ')[1: ]:
                splitted = elem.split(':')
                token = splitted[0]
                counter = int(float(splitted[1])) if len(splitted) > 1 else 1

                if token not in token2id:
                    token2id[token] = index
                    id2token[index] = token
                    id2counter[index] = counter

                    index += 1
                else:
                	id2counter[token2id[token]] += counter
                NNZ += 1

	with open(mm_vocab_filename, 'w') as fout:
		for i in xrange(1, 1 + len(id2token.keys())):
			fout.write('{}\t{}\t{}\n'.format(i, id2token[i], id2counter[i]))

	with open(vw_filename, 'r') as fin:
		with open(mm_data_filename, 'w') as fout:
			fout.write('%%' + 'MatrixMarket matrix coordinate real general\n')
			fout.write('{} {} {}\n'.format(D, len(token2id.keys()), NNZ))

			for doc_index, line in enumerate(fin):
				for elem in line.strip().split(' ')[1: ]:
					splitted = elem.split(':')
					token = splitted[0]
					counter = int(float(splitted[1])) if len(splitted) > 1 else 1

					fout.write('{} {} {}\n'.format(doc_index + 1, token2id[token], counter))


def train_test_split_mm_file(mm_filename, train_mm_filename, test_mm_filename, test_size=1e+5):
	NNZ_test = 0
	with open(mm_filename, 'r') as fin:
		header = fin.readline()
		D, W, NNZ = map(int, fin.readline().strip().split(' '))
		
		if test_size >= D:
			raise ValueError('Subsample size ({}) should be less than sample size ({})'.format(test_size, D))

		for line in fin:
			if int(line.split(' ')[0]) <= test_size:
				NNZ_test += 1
			else:
				break

		with open(train_mm_filename, 'w') as fout:
			fout.write(header)
			fout.write('{} {} {}\n'.format(D - test_size, W, NNZ - NNZ_test))

			def _write(row):
				splitted = row.strip().split(' ')
				fout.write('{} {} {}\n'.format(int(splitted[0]) - test_size, splitted[1], splitted[2]))

			_write(line)
			for line in fin:
				_write(line)

	with open(mm_filename, 'r') as fin:
		fin.readline()
		fin.readline()

		with open(test_mm_filename, 'w') as fout:
			fout.write(header)
			fout.write('{} {} {}\n'.format(test_size, W, NNZ_test))
			for i, line in enumerate(fin):
				if i < NNZ_test:
					fout.write(line)
				else:
					break
