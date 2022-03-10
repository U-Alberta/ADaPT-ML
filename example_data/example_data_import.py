"""
This is a Python script demonstrating how the datapoints for the example use case were prepared and loaded into CrateDB.
Pandas DataFrames have a method to_sql that makes it easy to create a table in CrateDB, but there are many other ways
that this can be accomplished. Please refer to CrateDB's documentation for more information.
"""
import pandas as pd
import spacy
import sqlalchemy
import tensorflow_hub as hub
from sqlalchemy.dialects import postgresql


def main():
    """
    defines example datapoints and loads them into a DataFrame (df), featurizes txt component for LFs and ML, imports
    the df into CrateDB, and "samples" datapoints from the df for the unlabeled training data dfs, then saves them.
    """
    use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")  # ML featurizer
    data = {
        'id': [
            # multilabel
            '00',  # just cat
            '01',  # just dog
            '02',  # just bird
            '03',  # cat and dog
            '04',  # cat and bird
            '05',  # dog and bird
            '06',  # cat, dog, and bird
            '07',  # lots of keywords for cat, dog, and bird
            '08',  # another cat and dog
            '09',  # just horse
            '10',  # just snake
            '11',  # horse and snake
            '12',  # horse and snake
            '13',  # bird, horse, and snake
            '14',  # none

            # multiclass
            '15',  # just cat
            '16',  # just cat
            '17',  # just cat
            '18',  # just dog
            '19',  # just dog
            '20',  # just dog
            '21',  # just bird
            '22',  # just bird
            '23',  # just bird
            '24',  # just bird
            '25',  # just horse
            '26',  # just horse
            '27',  # just snake
            '28',  # just snake
            '29'  # none
        ],
        'txt': [
            # multilabel
            "I hate cleaning my kitten's litterbox, but she's worth it!",
            "I took my husky to the park and he loved it.",
            "Its beak and feathers are very unique.",
            "The cat goes meow, and the dog goes woof.",
            "I hope that my tabby and parrot are going to be friends eventually.",
            "Barking at the little birds in the birdbath is his favourite thing to do.",
            "Tigers are ferocious, poodles bark and pant, and eagles have great talons and squawk a lot.",
            "whisker meow panther cheetah purr hiss bark woof bulldog howl puppy squawk feather chick.",
            "cheetah cougar leopard kitten cat schnauser collie labrador puppy dog.",
            "The pony excelled in agility.",
            "The snake slithered through the grass.",
            "The clydesdale was spooked by the serpent and galloped away.",
            "Horses have hooves and snakes have no legs at all.",
            "This bird was hissed at by the rattlesnake and made the stallion whinny.",
            "If there was a picture of any of these animals, a separate LF could be made to catch it.",

            # multiclass
            "This kitten is so cute!",
            "I encountered a cougar on my hike.",
            "The whiskers on lions and tigers help them move around.",
            "I have a husky that howls a lot.",
            "I want a pitbull puppy.",
            "My dog wants to play fetch all of the time.",
            "Chirp, chirp!",
            "There are some vulture chicks over there.",
            "Penguins cannot fly despite having wings.",
            "I took some photos of the eagle that is perched in the tree.",
            "Horses have hooves.",
            "neigh and whinny.",
            "My pet cobra is very majestic.",
            "If you hear a rattlesnake's tail, you'd better run away.",
            "This data point won't get any votes."
        ]
    }

    df = pd.DataFrame(data)
    nlp = spacy.load('en_core_web_sm')
    df['txt_clean_lemma'] = df['txt'].apply(lambda t: [token.lemma_ for token in nlp(t)])  # LF features
    df['txt_use'] = use_embed(data['txt']).numpy().tolist()  # ML feature vectors
    df['table_name'] = 'example_data'

    # import data into CrateDB
    df.to_sql('example_data', 'crate://localhost:4200', if_exists='replace', index=False,
              dtype={'txt_clean_lemma': postgresql.ARRAY(sqlalchemy.types.String),
                     'txt_use': postgresql.ARRAY(sqlalchemy.types.String)})

    # "sample" datapoints to create unlabeled training data for the multilabel setting
    multilabel_df = df.head(15).sample(frac=1).reset_index(drop=True)
    multilabel_df[['table_name', 'id']].to_pickle('./dp/unlabeled_data/multilabel_df.pkl')
    print(multilabel_df)

    # "sample" datapoints to create unlabeled training data for the multiclass setting
    multiclass_df = df.tail(15).sample(frac=1).reset_index(drop=True)
    multiclass_df[['table_name', 'id']].to_pickle('./dp/unlabeled_data/multiclass_df.pkl')
    print(multiclass_df)


if __name__ == '__main__':
    main()
