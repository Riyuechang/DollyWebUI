import os.path as os_path
import json

from websockets.sync.client import connect

from config import config
from tools.log import logger

#請求身份驗證
def AuthenticationRequest(
    pluginName: str, 
    pluginDeveloper: str, 
    Token_Dict: dict
) -> str:
    Token = Token_Dict["data"]["authenticationToken"]
    request_json = json.dumps({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "SomeID",
        "messageType": "AuthenticationRequest",
        "data": {
            "pluginName": pluginName,
            "pluginDeveloper": pluginDeveloper,
            "authenticationToken": Token
        }
    })

    return request_json

#請求身份驗證令牌
def AuthenticationTokenRequest(
    pluginName: str, 
    pluginDeveloper: str
) ->str:
    request_json = json.dumps({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "SomeID",
        "messageType": "AuthenticationTokenRequest",
        "data": {
            "pluginName": pluginName,
            "pluginDeveloper": pluginDeveloper,
            # "pluginIcon": "My Icon"
        }
    })

    return request_json

#請求快速鍵列表
def HotkeysInCurrentModelRequest():
    request_json = json.dumps({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "SomeID",
        "messageType": "HotkeysInCurrentModelRequest"
    })

    return request_json

#請求觸發快速鍵
def HotkeyTriggerRequest(hotkeyID: str):
    request_json = json.dumps({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "SomeID",
        "messageType": "HotkeyTriggerRequest",
        "data": {
            "hotkeyID": hotkeyID
        }
    })

    return request_json


#處理快速鍵列表
def hotkeyID_list_processing(response_Dict: dict) -> dict:
    availableHotkeys = response_Dict["data"]["availableHotkeys"]

    hotkey_correspondence_table = {}
    for hotkeyID_list in availableHotkeys:
        hotkey_correspondence_table[hotkeyID_list["name"]] = hotkeyID_list["hotkeyID"]

    return hotkey_correspondence_table

class Websocket_connect:
    def __init__(
        self,
        VTube_Studio_API_URL: str,
    ):
        self.websocket = connect(VTube_Studio_API_URL)
        self.hotkey_correspondence_table: dict

    #關閉
    def close(self):
        self.websocket.close()

    #向websocket傳送訊息
    def websocket_send(
        self,
        request: str
    ) -> dict:
        self.websocket.send(request)
        response = self.websocket.recv()
        logger.info(f"VTube Studio API訊息:{response}")
        response_Dict = json.loads(response)
        return response_Dict

    #連接VTube Studio API
    def Connect_to_VTube_Studio_API(
        self,
        pluginName: str, 
        pluginDeveloper: str
    ) -> bool:
        logger.info("正在嘗試連接VTube Studio")
        API_connection_status = False


        def apply_to_connect_to_VTube_Studio():
            nonlocal API_connection_status

            connection_request = self.websocket_send( #申請連線
                AuthenticationTokenRequest(
                    pluginName, 
                    pluginDeveloper
                ))

            if connection_request["messageType"] == "AuthenticationTokenResponse":
                logger.info("成功取得Authentication_Token")
                connection_confirmation = self.websocket_send( #用Authentication_Token連線到API
                    AuthenticationRequest(
                        pluginName, 
                        pluginDeveloper, 
                        connection_request
                    ))

                if connection_confirmation["data"]["authenticated"]: #連線成功
                    with open('Authentication_Token.json', 'w') as file: #保存Authentication_Token
                        json.dump(connection_request, file, indent=4)

                    API_connection_status = True
                    logger.info("已連接VTube Studio API")
                else:
                    logger.error("VTube Studio API連接失敗")
            else:
                logger.error("使用者拒絕連接VTube Studio API")


        if not os_path.isfile('Authentication_Token.json'):
            with open('Authentication_Token.json', 'w') as file:
                pass

            apply_to_connect_to_VTube_Studio()
            return API_connection_status

        with open('Authentication_Token.json', 'r') as file:
            Authentication_Token_json = json.load(file)
        
        restore_connection = self.websocket_send( #嘗試用上次的Authentication_Token恢復連接
            AuthenticationRequest(
                pluginName, 
                pluginDeveloper, 
                Authentication_Token_json
            ))

        if restore_connection["data"]["authenticated"]: #成功恢復連接
            API_connection_status = True
            logger.info("已連接VTube Studio API")
        else: #恢復連接失敗
            logger.info("恢復連接失敗")
            apply_to_connect_to_VTube_Studio()
        
        return API_connection_status
    
    #觸發快速鍵
    def hotkey_trigger(
        self, 
        sentiment_label
    ): 
        hotkey_name = config.VTube_Studio.sentiment_analysis[sentiment_label] #根據設定檔轉換成對應快速鍵名稱
        hotkeyID = self.hotkey_correspondence_table[hotkey_name] #根據快速鍵名稱轉換成快速鍵ID
        _ = self.websocket_send( #觸發對應快速鍵
            HotkeyTriggerRequest(hotkeyID)
        )

        logger.info(f"觸發VTube Studio快速鍵:{hotkey_name}")
