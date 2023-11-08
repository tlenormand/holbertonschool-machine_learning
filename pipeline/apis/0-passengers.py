#!/usr/bin/env python3
""" None """

import requests


def get_data(url):
    """ None """
    return requests.get(url).json()


def availableShips(passengerCount):
    """ None """
    data = get_data('https://swapi-api.hbtn.io/api/starships/')
    available_ships = []

    while data["next"]:
        for ship in data["results"]:
            passenger = ship["passengers"].replace(",", "")

            if passenger.isnumeric():
                n_passengers = int(passenger)

                if n_passengers >= passengerCount:
                    available_ships.append(ship["name"])

        data = get_data(data["next"])

    return available_ships
