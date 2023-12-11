from datetime import datetime, timedelta

import requests
import os


def get_news_material(query, time_span):
    """
    retrieves all headlines related to the query within a given time span
    available time spans: last 24 hours, last week, last month
    :param query: the keyword or phrase to look for
    :param time_span: how far to look back
    :return: a list of headlines related to the query
    """
    start_date = datetime.now()
    api_key = os.environ["API-KEY"]

    if time_span == "last 24 hours":
        end_date = datetime.now() - timedelta(hours=24)
    elif time_span == "last week":
        end_date = datetime.now() - timedelta(days=7)
    else:
        end_date = datetime.now() - timedelta(days=30)

    news_request = requests.get(
        url="https://newsapi.org/v2/everything",
        params={
            "apiKey": api_key,
            "q": query,
            "from_param": str(start_date),
            "to": str(end_date),
            "language": "en",
            "sort_by": "relevancy",
            "pageSize": 20,
            "page": 1
        }

    ).json()

    article_headlines = []

    if news_request["status"] == "ok" and news_request["totalResults"] > 0:
        for article in news_request["articles"][:20]:
            article_headlines.append(article["title"])
    return article_headlines


if __name__ == '__main__':
    print(get_news_material("tesla", "last week"))
