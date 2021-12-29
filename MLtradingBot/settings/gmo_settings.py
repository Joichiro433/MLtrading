import os
import configparser

conf = configparser.ConfigParser()
conf.read(os.path.join('settings', 'settings.ini'), encoding='utf=8')

api_key = conf['gmo']['api_key']
api_secret_key = conf['gmo']['api_secret_key']

symbol = conf['gmo']['symbol']