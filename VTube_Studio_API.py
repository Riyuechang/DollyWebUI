import json

from websockets.sync.client import connect

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

#向websocket傳送訊息
def websocket_send(
    websocket, 
    request: str
) -> dict:
    websocket.send(request)
    response = websocket.recv()
    logger.info(f"VTube Studio API訊息:{response}")
    response_Dict = json.loads(response)
    return response_Dict

#連接VTube Studio API
def Connect_to_VTube_Studio_API(
    websocket, 
    pluginName: str, 
    pluginDeveloper: str
) -> bool:
    logger.info("正在嘗試連接VTube Studio")
    API_connection_status = False

    with open('Authentication_Token.json', 'r') as file:
        Authentication_Token_json = json.load(file)
    
    restore_connection = websocket_send( #嘗試用上次的Authentication_Token恢復連接
        websocket,
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
        connection_request = websocket_send( #申請連線
            websocket,
            AuthenticationTokenRequest(
                pluginName, 
                pluginDeveloper
            ))

        if connection_request["messageType"] == "AuthenticationTokenResponse":
            logger.info("成功取得Authentication_Token")
            connection_confirmation = websocket_send( #用Authentication_Token連線到API
                websocket,
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
    
    return API_connection_status
