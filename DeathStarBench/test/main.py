from pymongo import MongoClient
import time
print("Testing...")
client = MongoClient("geo-db:27017")
print(client)
db = client.admin
serverStatusResult=db.command("serverStatus")
print(serverStatusResult)
print("Done!")
time.sleep(60)
