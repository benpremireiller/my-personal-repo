import praw
import psycopg2
import pandas as pd
from psycopg2 import Error
import sys
sys.path.append('\\.')
from user_variables import client_id, client_secret, user_agent, password

def create_reddit_instance():
    reddit = praw.Reddit(client_id= client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)
    return reddit

def pull_submissions_by_tag(sub_reddit, tag, limit):

    reddit = create_reddit_instance()

    #Filter subreddit for tag
    if tag == 'top':
        submissions = reddit.subreddit(sub_reddit).top(limit = limit)
    elif tag == 'hot':
        submissions = reddit.subreddit(sub_reddit).hot(limit = limit)
    elif tag == 'new':
        submissions = reddit.subreddit(sub_reddit).new(limit = limit)
    else:
        raise ValueError('Tag must be set to top, hot or new')

    submission_data = []

    #Extract needed data from top 'limit' submissions
    for submission in submissions:

        submission = clean_submission(submission)
        submission_data.append((submission['created_utc'],
                                submission['selftext'],
                                submission['link_flair_text'],
                                sub_reddit,
                                submission['id'],
                                submission['title']
                                ))

    return submission_data

def clean_submission(sub):

    submission = {
        'created_utc': pd.to_datetime(sub.created_utc, unit = 's'),
        'selftext': sub.selftext.replace('\n',''),
        'link_flair_text': str(sub.link_flair_text).lower(),
        'id': sub.id,
        'title': str(sub.title).lower(),
    }

    return submission

def extract_all_submissions(sub_reddit, limit = None):

    tags = ['top', 'hot']
    data = []

    for tag in tags:
        data += pull_submissions_by_tag(sub_reddit, tag, limit)
    
    data = list(set(data))

    return data

def load_to_db(data):
    try:
        connection = psycopg2.connect(user = "postgres",
                                    password = password,
                                    host = "127.0.0.1",
                                    port = "5432",
                                    database = "postgres")

        cursor = connection.cursor()

        #Delete all data
        delete_query = "DELETE FROM reddit.submissions;"
        cursor.execute(delete_query)

        #Insert new values into table
        insert_query = """INSERT into reddit.submissions
         (date, text, tag, sub_reddit, submission_id, title) values 
         (%s, %s, %s, %s, %s, %s)
        """
        cursor.executemany(insert_query, data)

        #Seal the deal
        connection.commit()

    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)

    finally:
        if(connection):
            cursor.close()
            connection.close()


#Run etl
data = extract_all_submissions('AmItheAsshole')
load_to_db(data)

