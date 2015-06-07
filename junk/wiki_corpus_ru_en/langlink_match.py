# -*- coding: utf-8 -*-
# Script to match russian and english interwikies
import re
import pickle

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
# Find page_id -> page_title mapping for cross-language wikipedia links
# * filename - unpacked langlinks file (http://dumps.wikimedia.org/enwiki/20141208/enwiki-20141208-langlinks.sql.gz)
# * to_lang - Language code of the target, in the ISO 639-1 standard (example: "en" or "ru", but without "" quotes )
def find_langlinks(filename, to_lang):
    __REGEX = re.compile(r"\(([0-9]*),'" + to_lang + r"','([^']+)'\)")
    page_id_to_target = {}
    target_to_page_id = {}
    with open(filename) as f:
        for line in f:
            if not line.startswith('INSERT INTO `langlinks` VALUE'):
                continue

            res = __REGEX.findall(line)
            for (page_id, target_page) in res:
                target_page_lower = target_page.replace('_', ' ')
#                 print target_page_lower
                if '_' in target_page_lower:
                    print target_page_lower
                page_id_to_target[int(page_id)] = target_page_lower
                target_to_page_id[target_page_lower] = int(page_id)
    return (page_id_to_target, target_to_page_id)

# Only keeps targets that are present as keys in target_to_page_id
def find_title(filename, target_to_page_id) :
    __REGEX = re.compile(r"\(([0-9]*),[0-9]*,'([^']+)',")
    map = {}
    with open(filename) as f:
        for line in f:
            if not line.startswith('INSERT INTO `page` VALUE'):
                continue

            res = __REGEX.findall(line)
            for (page_id, target_page) in res:
                target_page_lower = target_page.replace('_', ' ')
                if '_' in target_page_lower:
                    print target_page_lower
                if target_to_page_id.has_key(target_page_lower):
                    map[target_page_lower] = int(page_id)
    return map

if True:
    [ru2en, ru2en_target] = find_langlinks(r'E:\DATA_TEXT\Wiki\ruwiki-20150110-langlinks.sql', 'en')
    print "len(ru2en) = %i" % len(ru2en)
    save_obj(ru2en, 'ru2en')

    [en2ru, en2ru_target] = find_langlinks(r'E:\DATA_TEXT\Wiki\enwiki-20141208-langlinks.sql', 'ru')
    print "len(en2ru) = %i" % len(en2ru)
    save_obj(en2ru, 'en2ru')

    ru_page_id = find_title(r'E:\DATA_TEXT\Wiki\ruwiki-20150110-page.sql', en2ru_target)
    print "len(ru_page_id) = %i" % len(ru_page_id)
    save_obj(ru_page_id, 'ru_page_id')

    en_page_id = find_title(r'E:\DATA_TEXT\Wiki\enwiki-20141208-page.sql', ru2en_target)
    print "len(en_page_id) = %i" % len(en_page_id)
    save_obj(en_page_id, 'en_page_id')

ru_page_id_map = load_obj('ru_page_id')
print "len(ru_page_id2) = %i" % len(ru_page_id_map)

en_page_id_map = load_obj('en_page_id')
print "len(en_page_id) = %i" % len(en_page_id_map)

ru2en_map = load_obj('ru2en')
print "len(ru2en) = %i" % len(ru2en_map)

en2ru_map = load_obj('en2ru')
print "len(en2ru) = %i" % len(en2ru_map)

mutual_count = 0
filename = 'ru2en.csv'
ru_page_ids = {}
with open('ru2en.csv', 'w') as f:
    for en_page_id in en2ru_map.keys():
        ru_page_title = en2ru_map[en_page_id]
        if not ru_page_id_map.has_key(ru_page_title):
            continue
        ru_page_id = ru_page_id_map[ru_page_title]
        if not ru2en_map.has_key(ru_page_id):
            continue
        en_page_title = ru2en_map[ru_page_id]
        mutual_count = mutual_count + 1
        f.write(str(ru_page_id) + "|" + ru_page_title + "|" + str(en_page_id) + "|" + en_page_title + "\n")
        ru_page_ids[ru_page_id] = True
    save_obj(ru_page_ids, 'ru_page_ids')
    print str(mutual_count) + ' entries written to ' + filename
