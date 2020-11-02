from snorkel.preprocess.nlp import SpacyPreprocessor

spacy_preprocessor = SpacyPreprocessor('text', 'spacy_doc', language='en_core_web_lg', memoize=True)
