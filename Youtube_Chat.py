import re
import sys
import atexit
from argparse import ArgumentParser

import pytchat
import requests

from loguru import logger

#設定log輸出格式
logger.remove() #清空預設值
fmt = "{level}|{message}"
logger.add(sys.stdout, format=fmt)

#傳遞參數
parser = ArgumentParser()
parser.add_argument("-c", "--channelid", help="YouTube帳號代碼", dest="channelid", default="", type=str)
args = parser.parse_args()

#提取直播的影片代碼
youtube_channel_id = args.channelid

if not re.search(r"@", youtube_channel_id):
    youtube_channel_id = "@" + youtube_channel_id

youtube_url = f"https://www.youtube.com/{youtube_channel_id}/live"
youtube_html = requests.get(youtube_url)

if youtube_html.status_code == 200:
    youtube_html_content = youtube_html.content.decode()
    video_id_search = re.search(r"(?<=https://www\.youtube\.com/watch\?v=)[a-zA-Z0-9-_]+", youtube_html_content)

    if video_id_search:
        video_id = video_id_search.group(0)
        logger.info("已連接上YouTube直播")

        try:
            YouTube_chat = pytchat.create(video_id=video_id) #讀取直播

            def Execute_at_the_end():
                YouTube_chat.terminate() #停止獲取聊天室

            atexit.register(Execute_at_the_end) #註冊關閉事件

            #讀取聊天室
            while YouTube_chat.is_alive():
                for response in YouTube_chat.get().sync_items(): #等待新訊息
                    data = {"author_name" : response.author.name, "message" : response.message} #輸出成指定格式
                    logger.info(data)
        except:
            logger.error("載入YouTube聊天室時發生錯誤")
    else:
        logger.error("YouTube直播未開啟")
else:
    logger.error("無法連線到YouTube直播")