# import gossipcat as gc 
# import pandas as pd 
# import numpy as np 
# import lightgbm as lgb 
# from sklearn import tree

# #DataSet
# #[size,weight,texture]
# X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],[166, 65, 40],
#      [190, 90, 47], [175, 64, 39],
#      [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# Y = ['apple', 'apple', 'orange', 'orange', 'apple', 'apple', 'orange', 'orange',
#      'orange', 'apple', 'apple']

# #classifier - DecisionTreeClassifier
# clf_tree = tree.DecisionTreeClassifier()
# clf_tree = clf_tree.fit(X,Y)

# #test_data
# test_data = [[190,70,42],[172,64,39],[182,80,42]]

# #prediction
# prediction_tree = clf_tree.predict(test_data)

# print("Prediction of DecisionTreeClassifier:",prediction_tree)

from flask import Flask
from redis import Redis, RedisError
import os
import socket
import gossipcat

# Connect to Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

app = Flask(__name__)

@app.route("/")
def hello():
    try:
        visits = redis.incr("counter")
    except RedisError:
        visits = "<i>cannot connect to Redis, counter disabled</i>"

    html = "<h3>Hello {name}!</h3>" \
    	   "<h3>This is a test!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>" \
           "<b>Visits:</b> {visits}"
    return html.format(name=os.getenv("NAME", "GossipCat"), hostname=socket.gethostname(), visits=visits)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)