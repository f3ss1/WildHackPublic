# WildHack 2021

This repo contains our project for the [Wildberries Hackathon 2021](https://hack-app.wildberries.ru/contests/hack#/).

## Task 2: Searching tags

> Implement an algorithm of recommendation hints, which reduces the search for the desired products in the system.

## Prerequirements

- [`Natasha`](https://github.com/natasha/natasha/)
- for [`Navec`](https://github.com/natasha/navec/) setup we use this [model](https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar)
- [`python-telegram-bot`](https://python-telegram-bot.org/)

## Project structure

- `embeddings_prep.py` is used for lemmatizing text words and mapping to their embedding representation. We
  use [`Natasha`](https://github.com/natasha/natasha/)
  library.
- `popularity.py` is used for mapping words to their popularity.
- `bot.py` is used for launching bot.
- `metrics.py` is our small file to separate the cosine similarity metric we use from the rest of the project.

## Launch order

1. `pip install requirements.txt` in order to install all the required libraries.
1. Launch `popularity.py` to create `popularity.pkl` containing information about possible tags' relative 'popularity'.
1. Launch `embeddings_prep.py` to create `embeddings.pkl` containing information about possible tags' embeddings.
1. Launch `bot.py` and input your bot token to launch Telegram bot with your token which would allow you to get tags for you queries based on results of previous steps.

However, this repo also contains our pretrained version of the solution (the required for `bot.py` `.pkl` files) so it can be used out-of-box by installing required dependances, setuping Navec and simply running `bot.py`. Still, feel free to go into both `embeddings_prep.py` and `popularity.py` to try to tune some parameters.