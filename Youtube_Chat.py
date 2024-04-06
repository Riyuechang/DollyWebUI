import sys
import atexit
import pytchat
from argparse import ArgumentParser
from loguru import logger

logger.remove()
fmt = "{message}"
logger.add(sys.stdout, format=fmt)

parser = ArgumentParser()
parser.add_argument("-v", "--videotoken", help="YouTube直播的影片代碼", dest="videotoken", default="")
args = parser.parse_args()

try:
    YouTube_chat = pytchat.create(video_id=args.videotoken)

    def Execute_at_the_end():
        YouTube_chat.terminate()

    atexit.register(Execute_at_the_end)

    while YouTube_chat.is_alive():
        for response in YouTube_chat.get().sync_items():
            data_dict = {"author_name" : response.author.name, "message" : response.message}
            logger.info(data_dict)
except:
    logger.info("載入YouTube聊天室時發生錯誤")