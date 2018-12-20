import configparser

config = configparser.ConfigParser()

import os

dirname = os.path.dirname(__file__)


def get_firebase_storage():
    pass

def get_config(pair_name, interval, prefix='default'):
    prefix += '.'
    config.read("{}/{}{}.conf".format(dirname, prefix, pair_name))
    section_name = '{}-m'.format(interval)
    section = {}
    if section_name in config:
        section = config[section_name]
    else:
        section = config['any-m']

    section['pair_name'] = config['DEFAULT']['pair_name']
    return section, config
