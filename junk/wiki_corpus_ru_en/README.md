Parallel bag-of-words corpus of Wikipedia pages on two languages: English and Russian
=====================================================================================

Загрузка данных
---------------

**EN wiki (latest)**

* https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2  - articles
* https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-langlinks.sql.gz - cross-language links

**RU wiki (latest)**

* https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2  - articles
* https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-langlinks.sql.gz  - cross-language links

Трюки с gensim
--------------

* Подправить https://github.com/piskvorky/gensim/blob/develop/gensim/corpora/wikicorpus.py вот так:

```
#===============================================================================
#         #for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
#             #for tokens, title, pageid in pool.imap(process_article, group): # chunksize=10):
#                 ... // continue with processing tokens
# =>
#         for text in texts:
#                 tokens, title, pageid = process_article(text) # chunksize=10):
#                 ... // continue with processing tokens
#===============================================================================
```

Описание схемы запуска эксперимента от Саши Фрея.
-------------------------------------------------

#### Шаг первый
запарсить файлы с кросс-язычными ссылками

`langlink_match.py` - для поиска пар, на выходе - `ru2en.csv`

#### Шаг второй
простанировать оба дампа, и сохранить в батчи только те документы, которые встречаются в `ru2en.csv`

`wikicorpus_process.py`, на выходе - две папки с батчами (русская и английская)

#### Шаг третий - стемминг русских батчей:
`russian_stemming.py`

Делает еще одну папку с обработанными батчами


#### Удаление частых/редких слов
Мы с Мариной вручную обрезали батчи (удаляли слишком частые и слишком редкие слова. Сейчас это проще возложить на ядро библиотеки. Хоят возможно это понадобиться, если на следующем шаге тебе не хватит памяти чтобы соединить батчи.

`get_dictionary.py`, `cut_batches.py`

#### Шаг четвертый
соединяем батчи. Грузит всё в памяти! Мне было нужно >16 GB.

`wikicorpus_merge.py`

#### Шаг последний
запускаемся :)

`multimodal_offline.py`

#### Глазами посмотреть на батчи
`print_items.py`
