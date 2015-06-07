import artm.messages_pb2, artm.library, time

#Download data from https://s3-eu-west-1.amazonaws.com/artmbigfiles/merged_en_ru_batches.7z
batches_folder = 'C:\\Users\\Administrator\\Documents\\GitHub\\latex\\vfar14bigartm\\merged_batches\\'

numTopics = 20     # |T| - overall number of tokens
numProcessors = 3   # This value defines how many concurrent processors to use for calculation.
numInnerIters = 10  # Typical values of this parameter are between 10 and 50. The larger it is the better for
                    # convergence, but large values will increase runtime proportionally to this parameter.
numTopTokensToShow = 12     # how many top-N tokens to visualize on each iteration
num_outer_iterations = 10   # how many outer iterations to perform (non-online algorithm)

#unique_tokens = artm.library.Library().LoadDictionary(train_batches_folder + 'dictionary')

master_config = artm.messages_pb2.MasterComponentConfig()
master_config.processors_count = numProcessors
master_config.disk_path = batches_folder

perplexity_collection_config = artm.messages_pb2.PerplexityScoreConfig()
perplexity_collection_config.model_type = artm.library.PerplexityScoreConfig_Type_UnigramDocumentModel
#perplexity_collection_config.model_type = artm.library.PerplexityScoreConfig_Type_UnigramCollectionModel
#perplexity_collection_config.dictionary_name = unique_tokens.name

with artm.library.MasterComponent(master_config) as master:
    #dictionary = master.CreateDictionary(unique_tokens)

    perplexity_score = master.CreatePerplexityScore(config = perplexity_collection_config)
    items_processed_score = master.CreateItemsProcessedScore()
    en_top_tokens = master.CreateTopTokensScore(num_tokens=numTopTokensToShow, class_id="@english")
    ru_top_tokens = master.CreateTopTokensScore(num_tokens=numTopTokensToShow, class_id="@russian")

    # Configure the model
    model_config=artm.messages_pb2.ModelConfig()
    model_config.class_id.append("@russian")
    model_config.class_id.append("@english")
    model = master.CreateModel(config=model_config, topics_count=numTopics, inner_iterations_count=numInnerIters)
    model.EnableScore(items_processed_score)
    model.EnableScore(perplexity_score)
    #model.Initialize(dictionary)    # Set random (but deterministic) initial approximation for Phi matrix

    for i in range(0, num_outer_iterations):
        start_time = time.time()
        master.InvokeIteration(1)       # Invoke one scan of the entire collection
        master.WaitIdle()
        model.Synchronize()
        print "Iteration time : %.3f " % (time.time() - start_time),
        print "Items processed = %i" % items_processed_score.GetValue(model = model).value,
        print "Train perplexity = %.3f" % perplexity_score.GetValue(model = model).value   # (?) perplexity score gives stupid garbage for multimodal collections

        artm.library.Visualizers.PrintTopTokensScore(en_top_tokens.GetValue(model))
        artm.library.Visualizers.PrintTopTokensScore(ru_top_tokens.GetValue(model))
