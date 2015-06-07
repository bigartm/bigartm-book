import glob, artm.messages_pb2, artm.library, os

def cut_batch(batch_path, batch_cut_path, dictionary_path, no_below=5, no_above=0.2):
    print "Loading " + dictionary_path + "..."
    dictionary = artm.library.Library().LoadDictionary(dictionary_path)
    print "Loading " + dictionary_path + " done, " + "|W|=" + str(len(dictionary.entry)) + ",",
    print "|D|=" + str(dictionary.total_items_count)

    dic_token_to_dic_entry = dict()
    for dic_entry_id, dic_entry in enumerate(dictionary.entry):
        if (no_below <= dic_entry.items_count) and (dic_entry.items_count <= (dictionary.total_items_count * no_above)):
            dic_token_to_dic_entry[dic_entry.class_id + dic_entry.key_token] = dic_entry

    processed = 0
    all_batch_list = glob.glob(os.path.join(batch_path,'*.batch'))
    for batch_file in all_batch_list:
        processed += 1
        print "Loading " + batch_file + " (" + str(processed) + " of " + str(len(all_batch_list)) + ")..."
        batch = artm.library.Library().LoadBatch(batch_file)
        print "Loading " + batch_file + " done, |W|=" + str(len(batch.token)) + ", |D|=" + str(len(batch.item))
        batch_cut = artm.messages_pb2.Batch()
        batch_cut_token_to_token_id = dict()
        lang_stat = dict()
        lang_stat["@english"] = 0
        lang_stat["@russian"] = 0
        for item in batch.item:
            item_cut = batch_cut.item.add()
            item_cut.title = item.title
            for field in item.field:
                field_cut = item_cut.field.add()
                field_cut.name = field.name
                for token_num in xrange(len(field.token_id)):
                    token_id = field.token_id[token_num]
                    token_count = field.token_count[token_num]
                    token = batch.token[token_id]
                    class_id = batch.class_id[token_id]
                    if (class_id + token) in dic_token_to_dic_entry:
                        if (class_id + token) not in batch_cut_token_to_token_id:
                            batch_cut.token.append(token)
                            batch_cut.class_id.append(class_id)
                            batch_cut_token_to_token_id[class_id + token] = (len(batch_cut.token) - 1)
                            lang_stat[class_id] += 1

                        field_cut.token_id.append(batch_cut_token_to_token_id[class_id + token])
                        field_cut.token_count.append(token_count)
        #save batch_cut
        print "Saving cut batch for " + batch_file + "..."
        artm.library.Library().SaveBatch(batch_cut, batch_cut_path)
        print "Saving cut batch for " + batch_file + " done,",
        print "|W|=" + str(len(batch_cut.token)) + ", |D|=" + str(len(batch_cut.item))
        print lang_stat

batches_folder = '/home/ubuntu/sashafrey/latex/vfar14bigartm/merged_batches/'
batches_folder_cut = '/home/ubuntu/sashafrey/latex/vfar14bigartm/merged_batches_cut/'

#batches_folder = '/home/ubuntu/sashafrey/latex/vfar14bigartm/3batches/'
#batches_folder_cut = '/home/ubuntu/sashafrey/latex/vfar14bigartm/3batches_cut/'

dictionary_path = os.path.join(batches_folder, 'dictionary')
cut_batch(batches_folder, batches_folder_cut, dictionary_path, no_below=20, no_above=0.1)
