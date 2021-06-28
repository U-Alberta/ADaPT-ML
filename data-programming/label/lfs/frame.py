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



frames = {
    "SS": [
        # column 1
        # "a generic frame that exists for other techno-scientific issues",
        # "historic and persistent for climate change",
        # "assumes a linear model of science into policymaking",
        # "political actors may use frame to focus attention onto the climate science away from addressing political "
        # "realities",
        # "increasing general scientific knowledge unlikely to alter engagement",
        # column 2
        # "emphasis on the science of climate change across any working_group",
        "there is broad expert scientific consensus",
        "considerable evidence of the need for action",
        "science has spoken",
        "politicians must act in terms of global agreements",
        # "any mention of uncertainty or scepticism quashed",
        # column 3
        # "scientists especially ipcc chair and co-chairs",
        # "sceptics absent or very much minority voices",
        # "politicians john_kerry john_p_holdren william_hague emphasize scientific consensus and the need to act now",
        # column 4
        "exhaustive IPCC report produced by thousands of expert scientists",
        "unprecedented rate of change compared to paleo records",
        "carbon budget emissions allowance in order to meet 2 degrees celsius policy target",
        "severe and irreversible impacts",
        "trust climate scientists and dismiss skeptic voices",
        # column 5
        "settled science",
        "unequivocal nature of anthropogenic climate change",
        "landmark report by IPCC",
        "the balance of evidence",
        "what more proof do we need",
        "greatest challenge of our time",
        "skeptics wishful thinking or malpractice",
        # column 6
        # "scientists undertaking fieldwork or in ipcc event",
        # "climate impacts flooding or places threatened by climate change kiribati",
        # "graphs and figures of climate science"
        # CUSTOM FRAMES
        "go read the IPCC report",
        "citing sources of information"
    ],
    "US": [
        # column 1
        # "this frame assumes a linear relationship between scientific evidence and policymaking",
        "there is still a lack of scientific evidence to justify action",
        # "newsroom routines may lead to this frame through journalists seeking to create a balanced news report",
        # column 2
        "uncertainty in climate science impacts or solutions",
        "question anthropogenic nature of climate change",
        "natural variability",
        "science has been wrong before and still lacks knowledge",
        "we cannot should not or will struggle to act",
        # column 3
        # "may be duelling experts but often sceptics are unchallenged",
        # "typical voices countering climate scientists include myron_ebell bob_carter benny_peisner",
        # column 4
        "unexplained pause in global mean temperature warming",
        "climatic research unit stolen emails",
        "climategate",
        "errors in IPCC",
        # "where climate scientists are present they attempt to counter the above by making appeals to expertise "
        # "evidence and observations",
        # column 5
        "a pause in warming or slowdown",
        "we cannot be sure despite scientists best efforts",
        "scientists making errors or mistakes",
        # "emotive language concerning behaviour of scientists",
        "hysteria and silliness",
        "scientists admit or insist or are puzzled",
        "scientists attempt to prove climate change",
        "global warming believers",
        # column 6
        # "hackneyed cliched images polar or glacial scenes especially images of polar bears",
        # "scientists debating with sceptics"
        #CUSTOM
        "climate change hoax"
    ],
    "PIS": [
        # column 1
        # "a generic frame",
        # "in the context of the climate change issue it comments on rather than doing straight up and down reporting of "
        # "the ipcc",
        # "can lead to polarization of audiences if highly partisan",
        # column 2
        "a political or ideological conflict over the way the world should work",
        "conflict over solutions or strategy to address issues",
        "a battle for power between nations groups or personalities",
        # column 3
        # "politicial figures al_gore ed_davey baroness_worthington",
        # "thinktank",
        # "non-profit actors nigel_lawson bjorn_lomborg",
        # column 4
        "detail of specific policies",
        "green new deal",
        "climate change act",
        "disagreement over policies and policy detail",
        "questioning the motives or funding of opponents",
        # column 5
        "a battle or war or fierce debate of ideas",
        "government strategy confused",
        "how can the other political side ignore these scientific truths and not act"
        # "nations given personalities the united_states as the bad boy of the climate debate",
        # column 6
        # "political figures",
        # "images of climate protest",
        # "political cartoons"
        # CUSTOM
    ],
    "D": [
        # column 1
        # "appeals to journalistic values especially in terms of personalisation by linking to impacts of extremes on "
        # "people",
        # "compelling visual imagery to accompany fearful narratives",
        # "fearful narratives can lead to denial or apathy",
        # column 2
        "predicted impacts are dire with severe consequences",
        "impacts are numerous and threaten all aspects of life",
        "impacts will get worse and we are not well prepared",
        # column 3
        # "scientists",
        # "non-governmental_organization officials",
        # "local people affected by climate impacts",
        # column 4
        "unprecedented rise in global average surface temperature",
        "sea level rise",
        "snow and ice decline",
        "decline in coral reefs",
        "extreme weather including droughts heatwaves floods",
        "scale of the challenge is overwhelming",
        # column 5
        "positively frightening",
        "unnatural weather",
        "weather on steroids",
        "violent or extreme weather",
        "runaway climate change",
        "life is unsustainable",
        # column 6
        # "climate impacts of all types",
        "threatened species or ecosystems",
        "disaster-stricken people",
        # "scientific figures maps graphs infographics of climate impacts"
        # CUSTOM FRAMES
        "entire ecosystems are collapsing"
    ],
    "O": [
        # O1
        # column 1
        # "a generic frame for techno-scientific issues social progress",
        # "only recently emerging for climate change",
        # "used by academics and left-leaning politicians",
        # column 2
        "climate change poses opportunities",
        "reimagine how we live",
        "further human development",
        "invest in co-benefits",
        # column 3
        # "chris_field ipcc co-chair",
        # column 4
        # "climate change has provided an opportunity to improve lives now as well as in the future",
        # column 5
        "climate change is rich with opportunity",
        "time for innovation or creativity",
        "improve lives now and in the future",
        # column 6
        # "no key images"
        # CUSTOM
        "take personal action",
        "change in lifestyle choices",
        "change diet go vegan or vegetarian",
        "eco-friendly and sustainable cities and management",
        "eco-friendly and sustainable lifestyle",
        "reduce carbon footprint",
        "adapt to challenges",
        "adaptation strategies",
        # O2
        # column 1
        "CO2 fertilization for agriculture",
        # column 2
        "beneficial impacts of changing climate",
        # "negative impacts ignored or dismissed",
        "no intervention needed",
        # column 3
        # "industries",
        # column 4
        "melting arctic will lead to opening up of shipping routes",
        "new trade opportunities",
        "increased agricultural productivity through increasing atmospheric CO2 fertilization",
        # column 5
        "opportunity to transform trade",
        "increased resource extraction"
        # column 6
        # "no key images"
    ],
    "E": [
        # E1
        # column 1
        # "given impetus through united_kingdom chancellor of the exchequer gordon_brown call for an economic analysis "
        # "of climate change and the resulting stern_review",
        # "frame reinvigorated by recent campaigning for divestment from the fossil fuel industry",
        # column 2
        "economic growth prosperity investments and markets",
        "high monetary costs of inaction",
        "the economic case provides a strong argument for action now",
        "divestment from fossil fuels like oil and gas",
        # column 3
        # "government experts",
        # "economic advisors",
        # "nicholas_stern",
        # column 4
        "cost of mitigating climate change is high but the cost will be higher if we do not act now",
        "action now can create green jobs",
        # column 5
        "economic growth and prosperity",
        "costs and economic estimates",
        "billions of dollars of damage in the future if no action is taken now",
        "it will not cost the world to save the planet",
        # column 6
        # "no key images"
        # E2
        # column 1
        # "an early proponent of this frame was bjorn_lomborg in the copenhagen_consensus",
        # "frame used by conservative politicians to justify inaction on climate",
        # column 2
        "high monetary costs of action",
        "action is hugely expensive or simply too costly in the context of other priorities",
        "scientific uncertainty",
        # column 3
        # "government experts",
        # "economic advisors",
        # column 4
        "United Nations is proposing climate plans which will damage economic growth",
        "action at home now is unfair as Annex II countries will gain economic advantage",
        # column 5
        "action will damage economic growth",
        "it is no time for panicky rearranging of the global economy",
        "killing industry",
        "imposing costly energy efficiency requirements"
        # column 6
        # "no key images"
    ],
    "ME": [
        # column 1
        # "a generic frame",
        # "can be used in the climate change context to attempt to reach non-engaged groups",
        # "can be problematic politically if there is a perceived credibility gap between rhetoric and action",
        # column 2
        "an explicit and urgent moral religious or ethical call for action",
        "strong mitigation and protection of the most vulnerable",
        # " a key sign of this frame in use is mention of god ethics morals or morality",
        # column 3
        # "religious moral ethical leaders and thinkers rowan_williams",
        # column 4
        "God ethics and morality",
        "climate change linked to poverty",
        "ending world hunger",
        "millennium development goal",
        # column 5
        "exert moral pressure",
        "degradation of nature",
        "ruining the planet or creation",
        "people or nations at the front line of climate change for the most vulnerable and already exposed",
        # column 6
        # "religious moral or ethical leaders"
        #CUSTOM
        "responsibility to protect nature",
        "there is no planet b",
        # ME2
        "globalist climate change religion"
    ],
    "ROS": [
        # column 1
        # "this frame gained particular cultural currency in 2009 after the stolen cru emails were reported climategate"
        # "when much media reporting focused on the process of conducting climate science",
        # "often used to support critique or highlight the nature of the ipcc process",
        # column 2
        "process or role of science in society",
        "how the IPCC works or does not",
        "transparency in funding",
        "awareness of science",
        "institutions involving scientists like the IPCC",
        "public opinion understanding and knowledge",
        # "journalists may act as knowledge arbiters",
        # column 3
        # "daily_mail journalist david_rose bbc head of editorial standards david_jordan",
        # "diplomats john_ashton",
        # "academics roger_pielke",
        # "government bodies uk science_and_technology_select_committee",
        # column 4
        "bias in media sources",
        "giving contrarians a voice",
        "not broadcasting diverse views",
        "IPCC is a leading institution",
        "politicisation of science",
        "IPCC is too conservative or alarmist",
        "detail how IPCC process works",
        "amount of time and space given to contrarians or skeptics in the media",
        # column 5
        "threats to free speech",
        "false balance",
        "balance as bias",
        "sexed up science",
        "belief in scientists as a new priesthood of the truth",
        # column 6
        # "climate contrarians or sceptics",
        # "scientific figures",
        # "beautiful natural scenes rainforest animals"
        # CUSTOM
        "misinformation and propaganda",
        "fake news media",
        "hidden agenda and mainstream narrative",
        "suppression of information",
        "conflict of interest"
    ],
    "S": [
        # column 1
        # "gained traction in 2003 when a united_states defence study was leaked to the press",
        # "increasing use by governments and senior politicians",
        # "this frame can militarize the climate debate with implications for state-society relations",
        # column 2
        "threat to human energy",
        "threat to water supply",
        "threat to food security",
        "threats to the nation state especially over migration",
        "conflict might be local but could be larger in scale and endanger many",
        # column 3
        # "pentagon cia united_states military leaders rear admiral neil_morisetti",
        # "academics neil_adger david_lobell",
        # "non-governmental_organization voices",
        # column 4
        "conflicts may occur between developed and developing countries",
        "conflict between nature and humans",
        "conflict between different stakeholders in developed nations",
        # column 5
        "climate change as a threat multiplier",
        "increase in instability volatility and tension",
        "fighting for water security",
        "a danger to world peace",
        # column 4
        # "military leaders",
        "impacts on security usually related to food drought or migration",
        # CUSTOM
        "armed forces preparing for war",
        "people are being displaced"
    ],
    "H": [
        # column 1
        # "the health impacts of climate change are highlighted by elite institutions such as the "
        # "world_health_organization united_states environmental_protection_agency and public_health_england",
        # "the lancet has published special issues on climate change and health",
        # "holds power for connecting individuals to an abstract global issue",
        # column 2
        "severe danger to human health",
        "deaths from malnutrition",
        "deaths from insect-borne diseases",
        "poor air quality",
        "urgent mitigation and adaptation required",
        # column 3
        # "non-governmental_organization voices unicef oxfam",
        # "academics jason_west",
        # column 4
        "vulnerability of Annex II countries",
        "vulnerability of children and elders to health impacts",
        "details of health impacts from climate change",
        # column 5
        "health wellbeing livelihoods and survival are compromised",
        # column 6
        # "health impacts people wearing facemasks to protect against air pollution"
        # CUSTOM
        "financial cost of impacts to human health",
        "mental health issues",
        "worsening environmental and air pollution",
        "climate change is a global problem and affects everyone"
    ]
}
