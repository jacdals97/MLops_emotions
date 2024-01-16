from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from emotions.predict_model import Predictor
import pandas as pd
from contextlib import asynccontextmanager

# Load the model
artifact = "best_model:v1"

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        # Load the ML model
        model = Predictor(artifact)
        yield
    finally:
        # Unload the ML model
        model = None

app = FastAPI(lifespan=lifespan)


html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Emotion prediction</title>
    </head>
    <body>
        <h1>Emotion prediction service </h1>
        <h2>Type in the text you wish to predict below, and press "Predict"!</h2>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Predict</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        preds = model.predict(data)
        preds = preds[0] # removes the list thing
        preds=pd.DataFrame.from_dict(preds,orient='index')
        preds=preds.T
        preds=preds.astype({'score': 'float'})
        preds=preds.round(3)
        preds.score=preds.score*100
        await websocket.send_text(f"Message text was: {data} | With predicted label: {preds.label[0]} | And a probability of: {preds.score[0]} %")

        

