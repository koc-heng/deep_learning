from week8_tool import catch_ptt_title
import time
board_list = [
    ("https://www.ptt.cc/bbs/baseball/index.html", "baseball"),
    ("https://www.ptt.cc/bbs/Boy-Girl/index.html", "Boy-Girl"),
    ("https://www.ptt.cc/bbs/c_chat/index.html", "c_chat"),
    ("https://www.ptt.cc/bbs/hatepolitics/index.html", "hatepolitics"),
    ("https://www.ptt.cc/bbs/Lifeismoney/index.html", "Lifeismoney"),
    ("https://www.ptt.cc/bbs/Military/index.html", "Military"),
    ("https://www.ptt.cc/bbs/pc_shopping/index.html", "pc_shopping"),
    ("https://www.ptt.cc/bbs/stock/index.html", "stock"),
    ("https://www.ptt.cc/bbs/Tech_Job/index.html", "Tech_Job")
]


for url, board_name in board_list:
    csv_name = f"ptt_{board_name}_stream.csv"
    print(f"\n[GO] catch: {board_name} → write: {csv_name}")
    catch_ptt_title(
        board_url = url,
        board_name = board_name,
        max_titles = 50000,
        output_csv = csv_name
    )
    print(f"[Finished] board {board_name} \n")