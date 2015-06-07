import glob, artm.messages_pb2, artm.library, os

def print_items(batch):
    for item in batch.item[1:10]:
        print item.title + " : ",
        for field in item.field:
            for (token_id, token_count) in zip(field.token_id, field.token_count):
                print batch.token[token_id] + "(" + str(token_count) + "),",
        print "\n",

#batch = artm.library.Library().LoadBatch("C:\\datasets\\merged_batches_cut.tar\\merged_batches_cut\\001d4cc8-885a-471c-a0f2-528e2021a795.batch")
#batch = artm.library.Library().LoadBatch("D:\\datasets\\multilang_wiki\\en_batches\\0a7046b8-dfd4-4f75-8cdb-21226ef048ff.batch")
#batch = artm.library.Library().LoadBatch("C:\\datasets\\merged_batches_cut_test\\001d4cc8-885a-471c-a0f2-528e2021a795.batch")

batch = artm.library.Library().LoadBatch(r"C:\datasets\cut\06e38098-a193-40b0-816d-30e87091a3b2.batch")
print_items(batch)
print '\n',
batch = artm.library.Library().LoadBatch(r"C:\datasets\merged\2c787cd3-40ec-4d4a-a7e1-6153010bd2af.batch")
print_items(batch)
print '\n',
batch = artm.library.Library().LoadBatch(r"C:\datasets\en\6925e594-cce2-4581-a0ba-cc5fb7b237cc.batch")
print_items(batch)
print '\n',
batch = artm.library.Library().LoadBatch(r"C:\datasets\ru\8b56a371-0dd9-4ae0-a31b-253521a076ef.batch")
print_items(batch)
print '\n',
batch = artm.library.Library().LoadBatch(r"C:\datasets\ru_stem\fd8a5bc2-eacd-496d-b4b1-5cb6f6f7dbac.batch")
print_items(batch)


