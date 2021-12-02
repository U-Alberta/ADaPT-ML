"""
Description of frames from Table S2 in "Supplimentary Information for Dominant Frames in Legacy and Social Media
Coverage of the IPCC Fifth Assessment Report"

Some description has been left out like mentions of other frames, punctuation removed, everything lowercased,
entities joined by _

ss          Settled Science
us          Uncertain (and contested) Science
pis         Political or Ideological Struggle
d           Disaster
o1 & o2     Opportunity
e1 & e2     Economic
me1 & me2   Morality and Ethics
ros         Role of Science
s           Security
h           Health

column 1    socio-political context of frame
column 2    problem definition, moral judgement, remedy
column 3    typical sources
column 4    themes or storylines
column 5    language, metaphors, phrases
column 6    visual imagery
"""
import json
import time
from sentence_transformers import SentenceTransformer

from redditscore.tokenizer import CrazyTokenizer
from nltk.tokenize import sent_tokenize
import os
import tensorflow as tf
from tweet_parser.tweet import Tweet
import pandas as pd
from sqlalchemy.dialects.postgresql import ARRAY
from crate.client.sqlalchemy.types import Object
from sqlalchemy.types import String, DateTime, Float
import numpy as np
from tqdm import tqdm
import tensorflow_hub as hub
import pickle
import emoji
import re

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.prod((10, 1050)))

use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
roberta_model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

rule_poli = re.compile(r'[pP][oO][lL][iI]')
rule_govt = re.compile(r'[gG][oO][vV][tT]')
rule_2c = re.compile(r"""\d+[cC]                |   # e.g. 2c
                         \d+\.\d+[cC]           |   # e.g. 1.5C
                         \d+º[cC]               |   # e.g. 2ºC
                         \d+\.\d+º[cC]              # e.g. 1.5ºc 
                         """, re.X)
rule_mdg = re.compile(r'[mM][dD][gG][sS]|[mM][dD][gG]')
rule_ipcc = re.compile(r'[iI][pP][cC][cC]')
rule_un = re.compile(r'\s[uU][nN]\s')
rule_who = re.compile(r'\s[W][H][O]\s')
extra_patterns = [(rule_2c, ' degree celsius '),
                  (rule_mdg, ' Millennium Development Goal '),
                  (rule_poli, ' politics '),
                  (rule_govt, ' government '),
                  (rule_ipcc, ' Intergovernmental Panel on Climate Change '),
                  (rule_un, ' United Nations '),
                  (rule_who, ' World Health Organization ')]
phrase_tokenizer = CrazyTokenizer(lowercase=False,
                                  keepcaps=True,
                                  hashtags='split',
                                  remove_punct=False,
                                  decontract=True,
                                  # extra_patterns=extra_patterns,
                                  twitter_handles='',
                                  urls='',
                                  whitespaces_to_underscores=False)


def phrase_tokenize(text):
    all_phrases = {}
    phrases = sent_tokenize(text)
    for i in range(len(phrases)):
        phrase = phrases[i]
        for pattern in extra_patterns:
            phrase = re.sub(pattern[0], pattern[1], phrase)
        tokens = phrase_tokenizer.tokenize(phrase)
        tokens = [token if not emoji.is_emoji(token) else token.strip(':').replace('_', ' ') for token in tokens]
        all_phrases['sentence_{}'.format(i)] = {'tokens': tokens, 'phrase': ' '.join(tokens)}
    return all_phrases


def check_array_size(a):
    size = len(a.encode('utf-8'))
    try:
        assert size < 32766
    except:
        print("size: ", size)


frame_list = [
                 'settled_science'] * 18 + [
                 'uncertain_science'] * 18 + [
                 'political_or_ideological_struggle'] * 11 + [
                 'disaster'] * 18 + [
                 'opportunity'] * 23 + [
                 'economic'] * 19 + [
                 'morality_and_ethics'] * 13 + [
                 'role_of_science'] * 24 + [
                 'security'] * 15 + [
                 'health'] * 13
element_id_list = list(range(18)) + list(range(18)) + list(range(11)) + list(range(18)) + list(range(23)) + \
                  list(range(19)) + list(range(13)) + list(range(24)) + list(range(15)) + list(range(13))
element_list = [
    # settled science
    "there is broad expert scientific consensus",
    "considerable evidence of the need for action",
    "science has spoken",
    "politicians must act in terms of global agreements",
    "exhaustive Intergovernmental Panel on Climate Change report produced by thousands of expert scientists",
    "unprecedented rate of change compared to paleo records",
    "carbon budget emissions allowance in order to meet 2 degrees celsius policy target",
    "severe and irreversible impacts",
    "trust climate scientists and dismiss skeptic voices",
    "settled science",
    "unequivocal nature of anthropogenic climate change",
    "landmark report by Intergovernmental Panel on Climate Change",
    "the balance of evidence",
    "what more proof do we need",
    "greatest challenge of our time",
    "skeptics wishful thinking or malpractice",
    "go read the Intergovernmental Panel on Climate Change report",
    "citing sources of information",
    # uncertain science
    "there is still a lack of scientific evidence to justify action",
    "uncertainty in climate science impacts or solutions",
    "question anthropogenic nature of climate change",
    "natural variability",
    "science has been wrong before and still lacks knowledge",
    "we cannot should not or will struggle to act",
    "unexplained pause in global mean temperature warming",
    "Climatic Research Unit stolen emails",
    "climategate",
    "errors in Intergovernmental Panel on Climate Change",
    "a pause in warming or slowdown",
    "we cannot be sure despite scientists best efforts",
    "scientists making errors or mistakes",
    "hysteria and silliness",
    "scientists admit or insist or are puzzled",
    "scientists attempt to prove climate change",
    "global warming believers",
    "climate change hoax",
    # political or ideological struggle
    "a political or ideological conflict over the way the world should work",
    "conflict over solutions or strategy to address issues",
    "a battle for power between nations groups or personalities",
    "detail of specific policies",
    "green new deal",
    "climate change act",
    "disagreement over policies and policy detail",
    "questioning the motives or funding of opponents",
    "a battle or war or fierce debate of ideas",
    "government strategy confused",
    "how can the other political side ignore these scientific truths and not act",
    # disaster
    "predicted impacts are dire with severe consequences",
    "impacts are numerous and threaten all aspects of life",
    "impacts will get worse and we are not well prepared",
    "unprecedented rise in global average surface temperature",
    "sea level rise",
    "snow and ice decline",
    "decline in coral reefs",
    "extreme weather including droughts heatwaves floods",
    "scale of the challenge is overwhelming",
    "positively frightening",
    "unnatural weather",
    "weather on steroids",
    "violent or extreme weather",
    "runaway climate change",
    "life is unsustainable",
    "threatened species or ecosystems",
    "disaster-stricken people",
    "entire ecosystems are collapsing",
    # opportunity
    "climate change poses opportunities",
    "reimagine how we live",
    "further human development",
    "invest in co-benefits",
    "climate change is rich with opportunity",
    "time for innovation or creativity",
    "improve lives now and in the future",
    "take personal action",
    "change in lifestyle choices",
    "change diet go vegan or vegetarian",
    "eco-friendly and sustainable cities and management",
    "eco-friendly and sustainable lifestyle",
    "reduce carbon footprint",
    "adapt to challenges",
    "adaptation strategies",
    "carbon dioxide fertilization for agriculture",
    "beneficial impacts of changing climate",
    "no intervention needed",
    "melting arctic will lead to opening up of shipping routes",
    "new trade opportunities",
    "increased agricultural productivity through increasing atmospheric carbon dioxide fertilization",
    "opportunity to transform trade",
    "increased resource extraction",
    # economic
    "economic growth prosperity investments and markets",
    "high monetary costs of inaction",
    "the economic case provides a strong argument for action now",
    "divestment from fossil fuels like oil and gas",
    "cost of mitigating climate change is high but the cost will be higher if we do not act now",
    "action now can create green jobs",
    "economic growth and prosperity",
    "costs and economic estimates",
    "billions of dollars of damage in the future if no action is taken now",
    "it will not cost the world to save the planet",
    "high monetary costs of action",
    "action is hugely expensive or simply too costly in the context of other priorities",
    "scientific uncertainty",
    "United Nations is proposing climate plans which will damage economic growth",
    "action at home now is unfair as Annex II countries will gain economic advantage",
    "action will damage economic growth",
    "it is no time for panicky rearranging of the global economy",
    "killing industry",
    "imposing costly energy efficiency requirements",
    # morality and ethics
    "an explicit and urgent moral religious or ethical call for action",
    "strong mitigation and protection of the most vulnerable",
    "God ethics and morality",
    "climate change linked to poverty",
    "ending world hunger",
    "Millennium Development Goal",
    "exert moral pressure",
    "degradation of nature",
    "ruining the planet or creation",
    "people or nations at the front line of climate change for the most vulnerable and already exposed",
    "responsibility to protect nature",
    "there is no planet B",
    "globalist climate change religion",
    # role of science
    "process or role of science in society",
    "how the Intergovernmental Panel on Climate Change works or does not",
    "transparency in funding",
    "awareness of science",
    "institutions involving scientists like the Intergovernmental Panel on Climate Change",
    "public opinion understanding and knowledge",
    "bias in media sources",
    "giving contrarians a voice",
    "not broadcasting diverse views",
    "Intergovernmental Panel on Climate Change is a leading institution",
    "politicisation of science",
    "Intergovernmental Panel on Climate Change is too conservative or alarmist",
    "detail how Intergovernmental Panel on Climate Change process works",
    "amount of time and space given to contrarians or skeptics in the media",
    "threats to free speech",
    "false balance",
    "balance as bias",
    "sexed up science",
    "belief in scientists as a new priesthood of the truth",
    "misinformation and propaganda",
    "fake news media",
    "hidden agenda and mainstream narrative",
    "suppression of information",
    "conflict of interest",
    # security
    "threat to human energy",
    "threat to water supply",
    "threat to food security",
    "threats to the nation state especially over migration",
    "conflict might be local but could be larger in scale and endanger many",
    "conflicts may occur between developed and developing countries",
    "conflict between nature and humans",
    "conflict between different stakeholders in developed nations",
    "climate change as a threat multiplier",
    "increase in instability volatility and tension",
    "fighting for water security",
    "a danger to world peace",
    "impacts on security usually related to food drought or migration",
    "armed forces preparing for war",
    "people are being displaced",
    # health
    "severe danger to human health",
    "deaths from malnutrition",
    "deaths from insect-borne diseases",
    "poor air quality",
    "urgent mitigation and adaptation required",
    "vulnerability of Annex II countries",
    "vulnerability of children and elders to health impacts",
    "details of health impacts from climate change",
    "health wellbeing livelihoods and survival are compromised",
    "financial cost of impacts to human health",
    "mental health issues",
    "worsening environmental and air pollution",
    "climate change is a global problem and affects everyone"
]
element_use = use_embed(element_list).numpy()
element_roberta = roberta_model.encode(element_list)
element_roberta_norm = tf.keras.utils.normalize(element_roberta, axis=-1, order=2)
frames = {
    "element_id": element_id_list,
    "frame": frame_list,
    "element_txt": element_list,
    "element_use": element_use.tolist(),
    "element_roberta": element_roberta.tolist(),
    "element_roberta_norm": element_roberta_norm.tolist()
}

element_df = pd.DataFrame(frames)
element_df.to_sql('frame_elements', 'crate://localhost:4200', if_exists='append', index=False, dtype={
    'element_use': ARRAY(Float),
    'element_roberta': ARRAY(Float),
    'element_roberta_norm': ARRAY(Float)
})

ids = []
table = []
split = []
created_at_datetime = []
screen_name = []
bio = []
txt = []
processed = []
use_embeddings = []
use_median = []
use_avg = []
roberta_embeddings = []
roberta_median = []
roberta_avg = []
roberta_embeddings_norm = []
roberta_median_norm = []
roberta_avg_norm = []

with open('sample.jsonl', 'r') as infile:
    for line in tqdm(infile, desc='tweets'):
        start_time = time.time()
        tweet_dict = json.loads(line)
        tweet = Tweet(tweet_dict)
        if tweet.user_entered_text != '' and tweet.lang == 'en':
            ids.append(tweet.id)
            table.append('climate_tweets')
            split.append('sample')
            created_at_datetime.append(tweet.created_at_datetime)
            screen_name.append(tweet.screen_name)
            bio.append(tweet.bio)
            txt.append(tweet.user_entered_text)
            p = phrase_tokenize(tweet.user_entered_text)
            processed.append(json.dumps(p))
            s = [p[s]['phrase'] for s in p]
            ue = use_embed(s).numpy()
            use_embeddings.append(np.array_repr(ue))
            # use_embeddings.append(ue.tolist())

            use_median.append(np.array_repr(np.median(ue, axis=0)))
            # use_median.append(np.median(ue, axis=0).tolist())

            use_avg.append(np.array_repr(np.average(ue, axis=0)))
            # use_avg.append(np.average(ue, axis=0).tolist())

            rob = roberta_model.encode(s)
            roberta_embeddings.append(np.array_repr(rob))
            # roberta_embeddings.append(rob.tolist())
            # print(rob.shape)

            rob_med = np.median(rob, axis=0)
            roberta_median.append(np.array_repr(rob_med))
            # roberta_median.append(rob_med.tolist())
            # print(rob_med.shape)

            rob_avg = np.average(rob, axis=0)
            # print(rob_avg.shape)
            roberta_avg.append(np.array_repr(rob_avg))
            # roberta_avg.append(rob_avg.tolist())

            rob_norm = tf.keras.utils.normalize(rob, axis=-1, order=2)
            # print(rob_norm.shape)
            roberta_embeddings_norm.append(np.array_repr(rob_norm))

            rob_med_norm = tf.keras.utils.normalize(rob_med, axis=-1, order=2).flatten()
            # print(rob_med_norm.shape)
            roberta_median_norm.append(np.array_repr(rob_med_norm))

            rob_avg_norm = tf.keras.utils.normalize(rob_avg, axis=-1, order=2).flatten()
            roberta_avg_norm.append(np.array_repr(rob_avg_norm))
            # print(rob_avg_norm.shape)

        if len(ids) == 50:
            tweets_df = pd.DataFrame({
                'id': ids,
                'table_name': table,
                'split': split,
                'created_at_datetime': created_at_datetime,
                'screen_name': screen_name,
                'bio': bio,
                'txt': txt,
                'txt_clean_sentences': processed,
                'txt_clean_use': use_embeddings,
                'use_median': use_median,
                'use_average': use_avg,
                'txt_clean_roberta': roberta_embeddings,
                'roberta_median': roberta_median,
                'roberta_average': roberta_avg,
                'txt_clean_roberta_norm': roberta_embeddings_norm,
                'roberta_norm_median': roberta_median_norm,
                'roberta_norm_average': roberta_avg_norm
            })
            tweets_df['txt_clean_roberta'].apply(check_array_size)
            tweets_df['txt_clean_roberta_norm'].apply(check_array_size)
            tweets_df.to_sql('climate_tweets', 'crate://localhost:4200', if_exists='append', index=False,
                             dtype={'created_at_datetime': DateTime,
                                    'txt_clean_sentences': Object})
            ids = []
            table = []
            split = []
            created_at_datetime = []
            screen_name = []
            bio = []
            txt = []
            processed = []
            sentences = []
            use_embeddings = []
            use_median = []
            use_avg = []
            roberta_embeddings = []
            roberta_median = []
            roberta_avg = []
            roberta_embeddings_norm = []
            roberta_median_norm = []
            roberta_avg_norm = []

        end_time = time.time()
        print("total time taken this loop: ", end_time - start_time)

    tweets_df = pd.DataFrame({'id': ids,
                              'table_name': table,
                              'split': split,
                              'created_at_datetime': created_at_datetime,
                              'screen_name': screen_name,
                              'bio': bio,
                              'txt': txt,
                              'txt_clean_sentences': processed,
                              'txt_clean_use': use_embeddings,
                              'use_median': use_median,
                              'use_average': use_avg
                              # 'txt_clean_roberta': roberta_embeddings,
                              # 'roberta_median': roberta_median,
                              # 'roberta_average': roberta_avg,
                              # 'txt_clean_roberta_norm': roberta_embeddings_norm,
                              # 'roberta_median_norm': roberta_median_norm,
                              # 'roberta_average_norm': roberta_avg_norm
                              })
    # tweets_df['txt_clean_roberta'].apply(check_array_size)
    # tweets_df['txt_clean_roberta_norm'].apply(check_array_size)
    # print(tweets_df)
    tweets_df.to_sql('climate_tweets', 'crate://localhost:4200', if_exists='append', index=False,
                     dtype={'created_at_datetime': DateTime,
                            'txt_clean_sentences': Object})
