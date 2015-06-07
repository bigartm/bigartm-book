import glob, artm.messages_pb2, artm.library, os


def get_dictionary(batch_path, dictionary_path):
    dictionary = artm.messages_pb2.DictionaryConfig()
    dic_token_to_dic_entry_id = dict()
    for batch_file in glob.glob(os.path.join(batch_path,'*.batch')):
        print "Loading " + batch_file + "..."
        batch = artm.library.Library().LoadBatch(batch_file)
        print "Loading " + batch_file + " done, |W|=" + str(len(batch.token)) + ", |D|=" + str(len(batch.item))
        for token_id in xrange(len(batch.token)):
            token = batch.token[token_id]
            class_id = batch.class_id[token_id]
            if (class_id + token) not in dic_token_to_dic_entry_id:
                dic_entry = dictionary.entry.add()
                dic_entry.key_token = token
                dic_entry.class_id = class_id
                dic_token_to_dic_entry_id[class_id + token] = dic_entry

        for item in batch.item:
            for field in item.field:
                for token_num in xrange(len(field.token_id)):
                    token_id = field.token_id[token_num]
                    token_count = field.token_count[token_num]
                    token = batch.token[token_id]
                    class_id = batch.class_id[token_id]
                    dic_entry = dic_token_to_dic_entry_id[class_id + token]
                    dic_entry.token_count += token_count
                    dic_entry.items_count += 1
                    dictionary.total_token_count += token_count
            dictionary.total_items_count += 1

    # write dictionary
    print "Saving " + dictionary_path + "..."
    with open(dictionary_path, 'wb') as binary_file:
        binary_file.write(dictionary.SerializeToString())
    print "Saving " + dictionary_path + " done, |W|=" + str(len(dictionary.entry)) + ",",
    print "|D|=" + str(dictionary.total_items_count) + ", total_token_count=" + str(dictionary.total_token_count)


batches_folder = '/home/ubuntu/sashafrey/latex/vfar14bigartm/merged_batches/'
#batches_folder = '/home/ubuntu/sashafrey/latex/vfar14bigartm/3batches'
get_dictionary(batches_folder, os.path.join(batches_folder, 'dictionary'))
