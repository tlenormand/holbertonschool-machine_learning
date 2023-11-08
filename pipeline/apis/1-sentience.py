#!/usr/bin/env python3
""" None """

import requests


def get_data(url):
    """ None """
    return requests.get(url).json()


def sentientPlanets():
    """ None """
    data = get_data("https://swapi-api.hbtn.io/api/species/")
    sentient_planets = []

    while data["next"]:
        for specie in data["results"]:
            if "sentient" in [specie["designation"], specie["classification"]]:
                if specie["homeworld"]:
                    planet = get_data(specie["homeworld"])
                    sentient_planets.append(planet["name"])

        data = get_data(data["next"])

    for specie in data["results"]:
        if "sentient" in [specie["designation"], specie["classification"]]:
            if specie["homeworld"]:
                planet = get_data(specie["homeworld"])
                sentient_planets.append(planet["name"])

    return sentient_planets
