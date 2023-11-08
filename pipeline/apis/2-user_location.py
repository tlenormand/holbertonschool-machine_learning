#!/usr/bin/env python3
""" None """

import requests
import sys
import time


def get_data(url):
    """ None """
    header = {"Accept": "application/vnd.github.v3+json"}

    return requests.get(url, params=header)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        data = get_data(sys.argv[1])

        if data.status_code == 404:
            print("Not found")
        elif data.status_code == 403:
            print("Reset in {} min".format(
                int((int(data.headers["X-Ratelimit-Reset"]) - time.time())
                    / 60)
            ))
        else:
            data = data.json()
            print(data["location"])
