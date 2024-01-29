from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi



uri = [MongoDB_Uri]
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

reviewsdb = client.reviews
reviewscoll = reviewsdb.reviews

reviewslist = {}

def changeRevFormat():
    for c, review in enumerate(reviewscoll.find()):
        reviewdic = {}
        for v,k in review.items():
            reviewdic[v]= k
        reviewslist[c] = reviewdic

    return reviewslist