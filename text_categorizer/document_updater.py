# Please optimize this function because it will be called for each document.
# If this code changes the class distribution and the training and test
# subsets have already been created before using this code, it may be
# important to force their regeneration in the configuration file to obtain
# valid statistics.
# This code should work for the training phase and for the prediction phase.
def initial_code_to_run_on_document(doc):
    #doc.analyzed_sentences = None # Indicates that this document should be completely ignored.
    return
