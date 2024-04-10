from flask import Flask, render_template, request
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

#獲得顯示用網頁
@app.route("/index")
def index():
    return render_template('index.html')

#顯示文字到網頁
@app.route("/text_to_display", methods=['GET'])
def text_to_display():
    data = request.args.get('data')
    socketio.emit('text_to_display', data)
    return "Transmission completed"

#累加文字到網頁
@app.route("/text_to_display_add", methods=['GET'])
def text_to_display_add():
    data = request.args.get('data')
    socketio.emit('text_to_display_add', data)
    return "Transmission completed"

#清空網頁
@app.route("/clear")
def clear():
    socketio.emit('clear')
    return "clear"

if __name__ == '__main__':
    socketio.run(app, debug=True, host='127.0.0.1', port=3840)