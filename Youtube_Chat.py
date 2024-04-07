import sys
import atexit
import pytchat
from argparse import ArgumentParser
from loguru import logger

#設定log輸出格式
logger.remove() #清空預設值
fmt = "{message}"
logger.add(sys.stdout, format=fmt)

#傳遞參數
parser = ArgumentParser()
parser.add_argument("-v", "--videotoken", help="YouTube直播的影片代碼", dest="videotoken", default="")
args = parser.parse_args()

try:
    YouTube_chat = pytchat.create(video_id=args.videotoken) #讀取直播

    def Execute_at_the_end():
        YouTube_chat.terminate() #停止獲取聊天室

    atexit.register(Execute_at_the_end) #註冊關閉事件

    #讀取聊天室
    while YouTube_chat.is_alive():
        for response in YouTube_chat.get().sync_items(): #等待新訊息
            data_dict = {"author_name" : response.author.name, "message" : response.message} #輸出成指定格式
            logger.info(data_dict) #必須使用loguru的logger功能來輸出訊息,不能用print
except:
    logger.info("載入YouTube聊天室時發生錯誤")