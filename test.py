import gspread
from datetime import datetime, timezone, UTC
import random
import json



gc = gspread.service_account(filename="factorysimleaderboard-credentials.json")
sh = gc.open("FactorySimLeaderboard")
worksheet = sh.worksheet("Scores")

rows = []
for i in range(3):
#get current time in ISO format
    current_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    config = {"config": {"1": {"rotation": 0.0, "posY": 0.899645136704466, "posX": 0.49831701739789536}, "2": {"rotation": 0.0, "posY": 0.8266230233794492, "posX": 0.7775071137865748}, "4": {"rotation": 0.0, "posY": 0.2096679013690674, "posX": 0.09733432455395072}, "5": {"rotation": 0.0, "posY": 0.9134973560982824, "posX": 0.17561218444622403}, "3": {"posX": 0.729718749180887, "rotation": 0.0, "posY": 0.5758333925862723}, "0": {"posX": 0.8545904034161141, "rotation": 0.0, "posY": 0.9289495976281237}, "6": {"rotation": 0.0, "posY": 0.24708181760163364, "posX": 0.7037934397762898}}, "creator": "Hendrik Unger"}
    #created_at,	problem_id,	fitness,	creator, algorithm,     config
    rows.append([current_time, "7", random.random(), "Testuser", "testscript", json.dumps(config)])

worksheet.append_rows(rows, value_input_option="USER_ENTERED")