import numpy as np
import pickle

from natasha import (
    Doc,
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    MorphVocab
)
from navec import Navec
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
from telegram import Bot as Bot_
from metrics import metric

PATH = 'navec_hudlit_v1_12B_500K_300d_100q.tar'  # Name of file for Navec

NAME = 'embeddings'
NAME_POP = 'popularity'

TOKEN = ...
INPUT = 0

# Natasha setup.

segm = Segmenter()
_emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(_emb)
morph_vocab = MorphVocab()


def query_to_noun(query: str) -> list[str]:
    doc = Doc(query.lower())

    doc.segment(segmenter=segm)

    doc.tag_morph(morph_tagger)

    res_arr = []
    for token in doc.tokens:
        if token.pos == 'NOUN':
            token.lemmatize(morph_vocab)
            res_arr.append(token.lemma)

    return res_arr


# Navec setup.

navec = Navec.load(PATH)

# Loading pretrained embedding vocab.

with open(NAME + '.pkl', 'rb') as f:
    embed_dict = pickle.load(f)

with open(NAME_POP + '.pkl', 'rb') as f:
    pop_dict = pickle.load(f)


def get_tags(request: str) -> str:
    nouns = query_to_noun(request)

    if not len(nouns):
        return f'В запросе \'{request}\' не найдено существительных.'

    request_vec = np.zeros(300)
    found = False
    sum_weights = 0
    for noun in nouns:
        if noun in navec:
            if noun in pop_dict:
                request_vec += navec[noun] * pop_dict[noun]
                sum_weights += pop_dict[noun]
            else:
                request_vec += navec[noun]
                sum_weights += 1
            found = True
    if not found:
        return f'В запросе \'{request}\' не найдено существительных с реализованными эмбеддингами.'

    request_vec /= sum_weights

    distances = {
        key: (metric(request_vec, vec) / (np.log(pop_dict[key] + 1) + 1) if key in pop_dict else metric(request_vec, vec))
        for key, vec in embed_dict.items()}
    distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

    req_keys = list(distances.keys())[1:11]
    return f'Потенциальные теги для запроса \'{request}\': {req_keys}'


class Bot:
    def __init__(self, token: str = TOKEN):
        self.token = token

    def start(self) -> None:
        self.bot = Bot_(token=self.token)
        self.updater = Updater(self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.request()

    def stop(self) -> None:
        self.updater.stop()

    def start_msg(self, update, _):
        self.user_id = update.message.from_user.id

        msg = 'Привет! Введи запрос, содержащий существительное, и я подскажу потенциальные теги ' \
              'для твоего запроса.'
        update.message.reply_text(msg)

        return INPUT

    def cancel_msg(self, update, _):
        msg = 'Определение тегов остановлено.'
        update.message.reply_text(msg)
        return ConversationHandler.END

    def tags_reply(self, update, _):
        msg = get_tags(update.message.text)
        update.message.reply_text(msg)
        return INPUT

    def request(self) -> None:
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', self.start_msg)],
            states={
                INPUT: [MessageHandler(Filters.text & ~Filters.command, self.tags_reply)],
            },
            fallbacks=[CommandHandler('cancel', lambda update, context: ConversationHandler.END)],
        )
        self.dispatcher.add_handler(conv_handler)
        self.updater.start_polling()


if __name__ == '__main__':
    bot = Bot()
    bot.start()
    _ = input()
    bot.stop()
