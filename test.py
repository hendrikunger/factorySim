import gspread
from datetime import datetime, timezone, UTC
import random



gc = gspread.service_account(filename="factorysimleaderboard-credentials.json")
sh = gc.open("FactorySimLeaderboard")
worksheet = sh.worksheet("Scores")

rows = []
for i in range(3):
#get current time in ISO format
    current_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    individual = "{-0.46084216120770133,0.3565872627555882,0.681860800927812,-0.1016739185091246,-0.8318027985174798,-0.06422719666412458,-0.6937001301831331,-0.4746699732715368,-0.831276933866309,0.46670843485339625,-0.22257866099219215,0.408166271180178,-0.7636564071679158,0.1030291710306208,0.060298744742663765}"
    #created_at,	problem_id,	fitness,	individual,	creator,	config
    rows.append([current_time, 7, random.random(), individual, "ungertest", "config2"])

worksheet.append_rows(rows, value_input_option="USER_ENTERED")