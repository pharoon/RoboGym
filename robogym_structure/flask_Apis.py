# flask_api.py
from flask import Flask, request, jsonify, Response , stream_with_context
from flask_cors import CORS
from GUIMain import initialize, train as train_model_func, list_models as list_models_func, delete as delete_model_func, test as Test_Model
from model_manager import manager as mm
from database import models as db  
# import sys
# import io
app = Flask(__name__)
CORS(app)  #Allows calls from Electron frontend

db.init_db()

@app.route("/")
def home():
    return "RoboGym Flask API is running!"

@app.post("/initialize")
def api_initialize():
    initialize()
    return jsonify({"status": "initialized"})

@app.get("/train")
def api_train():
    model_name = request.args.get("model_name")
    timesteps = request.args.get("timesteps")
    task_number = request.args.get("task_number")

    print("model name is now ", model_name, flush=True)
    print("timesteps is now ", timesteps, flush=True)
    print("task_number is now ", task_number, flush=True)

    if not model_name or not timesteps or not task_number:
        return "Missing parameters", 400

    try:
        timesteps = int(timesteps)
        task_number = int(task_number)
    except ValueError:
        return "Invalid parameter types", 400

    def event_stream():
        yield "data: yarab"
        try:
            yield from train_model_func(model_name, timesteps, task_number)
        except Exception as e:
            yield f"data: ❌ Error: {str(e)}\n\n"
            yield "event: end\ndata: done\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.get("/models")
def api_list_models():
    return jsonify(mm.list_models())

@app.post("/delete")
def api_delete_model():
    data = request.json
    model_name = data["model_name"]
    delete_model_func(model_name)
    return jsonify({"status": "deleted"})

@app.get("/test")
def api_test():
    model_name = request.args.get("model")
    task_name = request.args.get("task")
    episodes = request.args.get("episodes", "1")

    if not model_name or not task_name:
        return "Missing parameters", 400

    def event_stream():
        yield f"data: 🔧 Starting test for model={model_name}, task={task_name}, episodes={episodes}\n\n"
        model = mm.load_model(model_name)
        yield f"data: ✅ Loaded model\n\n"
        try:
            yield from Test_Model(model, episodes=int(episodes), task_name="pick_and_place")
        except Exception as e:
            yield f"data: ❌ test_model raised exception: {str(e)}\n\n"
        yield "event: end\ndata: done\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


    # def start_testing():
    #     event_stream()
        # try:
        #     yield "We should start now"
        #     model = mm.load_model(model_name)
        #     print("Modes is now ", model)
        #     # Test_Model(model, episodes=10, task_name="pick_and_place")
        # except Exception as e:
        #     print(" test_model raised exception:", e, flush=True)
        #     return jsonify({"error": "Test model execution failed"}), 400

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(port=5000)
