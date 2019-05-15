import json
import warnings
warnings.filterwarnings('ignore')


def getConfig():
    config = dict()
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config 
    except Exception as e:
        print('[CRITIAL] NO CONFIGURATION FILE FOUND!')
        raise e