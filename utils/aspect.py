import pandas as pd
import en_core_web_lg

nlp = en_core_web_lg.load()


def find_aspect_phrases(text):
    text_doc = nlp(text)
    assert text_doc.is_parsed, "Did not create dependency tree"
    for i in range(len(text_doc)):
        token = text_doc[i]

