# flask_api.py
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from GUIMain import initialize, train as train_model_func, list_models as list_models_func, delete as delete_model_func, test as Test_Model, upload_model, GetModelRewards, compareModels
from model_manager import manager as mm
from database import models as db  
from multiprocessing import Process
from functools import wraps
import jwt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

# Load environment variables from Credentials.env
load_dotenv('Credentials.env')

app = Flask(__name__)
CORS(app)  # Allows calls from Electron frontend

# Configure JWT
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')  # Change in production
app.config['JWT_EXPIRATION_DELTA'] = timedelta(hours=int(os.getenv('JWT_EXPIRATION_HOURS', '24')))

db.init_db()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Token is missing!'}), 401

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = db.get_user_by_id(data['user_id'])
            if not current_user:
                return jsonify({'message': 'Invalid token!'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401

        return f(current_user, *args, **kwargs)
    return decorated

@app.route("/")
def home():
    return "RoboGym Flask API is running!"

@app.route("/register", methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password') or not data.get('email'):
        return jsonify({'message': 'Missing required fields!'}), 400
    
    if db.get_user_by_username(data['username']):
        return jsonify({'message': 'Username already exists!'}), 400
    
    try:
        user = db.create_user(
            username=data['username'],
            email=data['email'],
            password=data['password']
        )
        return jsonify({
            'message': 'User created successfully!',
            'user_id': user.id,
            'username': user.username,
            'email': user.email
        }), 201
    except Exception as e:
        return jsonify({'message': f'Error creating user: {str(e)}'}), 500

@app.route("/login", methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing username or password!'}), 400
    
    user = db.get_user_by_username(data['username'])
    
    if not user or not user.check_password(data['password']):
        return jsonify({'message': 'Invalid username or password!'}), 401
    
    token = jwt.encode({
        'user_id': user.id,
        'username': user.username,
        'exp': datetime.utcnow() + app.config['JWT_EXPIRATION_DELTA']
    }, app.config['SECRET_KEY'], algorithm="HS256")
    
    return jsonify({
        'token': token,
        'user_id': user.id,
        'username': user.username,
        'email': user.email
    })

@app.post("/initialize")
@token_required
def api_initialize(current_user):
    initialize()
    return jsonify({"status": "initialized"})

@app.get("/train")
@token_required
def api_train(current_user):
    model_name = request.args.get("model_name")
    timesteps = request.args.get("timesteps")
    task_number = request.args.get("task_number")

    if not model_name or not timesteps or not task_number:
        return "Missing parameters", 400

    try:
        timesteps = int(timesteps)
        task_number = int(task_number)
    except ValueError:
        return "Invalid parameter types", 400

    # Create user-specific directories if they don't exist
    user_models_dir = f"trained_models/user_{current_user.id}"
    user_logs_dir = f"logs/user_{current_user.id}"
    os.makedirs(user_models_dir, exist_ok=True)
    os.makedirs(user_logs_dir, exist_ok=True)

    # Create a trained model record in the database
    try:
        trained_model = db.create_trained_model(
            name=model_name,
            model_path=f"{user_models_dir}/{model_name}.zip",
            algorithm=db.AlgorithmType.PPO,  # Default to PPO for now
            robotic_arm=db.RoboticArmType.KUKA_IIWA,  # Default to KUKA_IIWA for now
            user_id=current_user.id
        )
    except Exception as e:
        return jsonify({'message': f'Error creating model record: {str(e)}'}), 500

    start_time = time.time()
    current_timestep = 0

    def event_stream():
        nonlocal current_timestep
        yield "data: 🟢 Training started...\n\n"
        start_time = time.time()
        mean_reward = None

        try:
            for event in train_model_func(
                model_name=model_name,
                total_timesteps=timesteps,
                task_name=task_number,
                model_path=f"{user_models_dir}/{model_name}.zip"
            ):
                # Remove 'data: ' prefix and process log lines
                line = event.strip().removeprefix("data: ").strip()

                if line.startswith("REWARD_LOG::"):
                    try:
                        _, logged_model_name, reward_str, timestep_str = line.split("::")
                        mean_reward = float(reward_str)
                        current_timestep = int(timestep_str)

                        db.log_training(
                            model_name=logged_model_name,
                            mean_reward=mean_reward,
                            current_timestep=current_timestep,
                            user_id=current_user.id
                        )
                    except (ValueError, IndexError):
                        pass  # Ignore malformed reward logs

                yield event  # Pass the full SSE line back to the frontend

            # Save the final training session
            total_time = time.time() - start_time
            logs_path = f"{user_logs_dir}/{model_name}"

            db.create_train_session(
                model_id=trained_model.id,
                user_id=current_user.id,
                timesteps=timesteps,
                total_time=total_time,
                logs_path=logs_path,
                mean_reward=mean_reward
            )

        except Exception as e:
            yield f"data: ❌ Error: {str(e)}\n\n"
            yield "event: end\ndata: failed\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.get("/models")
@token_required
def api_list_models(current_user):
    # Get models for the current user from the database
    user_models = db.get_user_models(current_user.id)
    
    if not user_models:
        return jsonify([])
    
    # Format the response
    models_list = []
    for model in user_models:
        try:
            created_at = None
            if hasattr(model, 'created_at') and model.created_at is not None:
                try:
                    created_at = model.created_at.isoformat()
                except AttributeError:
                    pass

            model_data = {
                'id': model.id,
                'name': model.name,
                'algorithm': str(model.algorithm.value) if hasattr(model.algorithm, 'value') else None,
                'robotic_arm': str(model.robotic_arm.value) if hasattr(model.robotic_arm, 'value') else None,
                'created_at': created_at,
                'model_path': model.model_path
            }
            models_list.append(model_data)
        except AttributeError:
            # Skip models with missing attributes
            continue
    
    return jsonify(models_list)

@app.post("/upload")
@token_required
def api_upload_model(current_user):
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
        
    filePath = data.get("FilePath")
    modelName = data.get("ModelName")
    
    if not filePath or not modelName:
        return jsonify({"status": "error", "message": "Missing FilePath or ModelName"}), 400

    try:
        # Create user-specific directory if it doesn't exist
        user_models_dir = f"trained_models/user_{current_user.id}"
        os.makedirs(user_models_dir, exist_ok=True)
        
        # First upload the model file to user-specific directory
        target_path = f"{user_models_dir}/{modelName}.zip"
        upload_model(file_path=filePath, model_name=modelName, target_path=target_path)
        
        # Create database record for the uploaded model
        trained_model = db.create_trained_model(
            name=modelName,
            model_path=target_path,
            algorithm=db.AlgorithmType.PPO,  # Default to PPO for uploaded models
            robotic_arm=db.RoboticArmType.PANDA,  # Default to PANDA for uploaded models
            user_id=current_user.id
        )
        
        return jsonify({
            "status": "ok",
            "model_id": trained_model.id,
            "message": "Model uploaded and saved to database successfully"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error uploading model: {str(e)}"
        }), 500

@app.post("/getRewards")
@token_required
def api_get_rewards(current_user):
    try:
        data = request.get_json(silent=True)
        if not data or 'Model_Name' not in data:
            return jsonify({"status": "error", "message": "Model name is required"}), 400

        model_name = data["Model_Name"]
        
        # Get the model's training history from our database
        logs = db.fetch_logs(model_name, current_user.id)
        if not logs:
            return jsonify({"status": "error", "message": "No training data found for this model"}), 404

        # Start the visualization process with user-specific paths
        user_models_dir = f"trained_models/user_{current_user.id}"
        Process(target=GetModelRewards, args=(model_name, user_models_dir)).start()
        
        return jsonify({
            "status": "ok",
            "message": "Fetching rewards visualization",
            "data": logs
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error fetching rewards: {str(e)}"
        }), 500

@app.post("/compareModels")
@token_required
def api_compare_Models(current_user):
    try:
        data = request.get_json(silent=True)
        if not data or 'first_model' not in data or 'second_model' not in data:
            return jsonify({"status": "error", "message": "Both model names are required"}), 400

        first_model = data["first_model"]
        second_model = data["second_model"]

        # Verify both models belong to the user
        user_models = db.get_user_models(current_user.id)
        user_model_names = [model.name for model in user_models]
        
        if first_model not in user_model_names or second_model not in user_model_names:
            return jsonify({
                "status": "error",
                "message": "One or both models not found or you don't have permission to access them"
            }), 404

        # Get training history for both models
        first_logs = db.fetch_logs(first_model, current_user.id)
        second_logs = db.fetch_logs(second_model, current_user.id)

        if not first_logs or not second_logs:
            return jsonify({
                "status": "error",
                "message": "Training data not found for one or both models"
            }), 404

        # Start comparison visualization with user-specific paths
        user_models_dir = f"trained_models/user_{current_user.id}"
        Process(target=compareModels, args=(
            first_model,
            second_model,
            user_models_dir
        )).start()

        return jsonify({
            "status": "ok",
            "message": "Comparing models",
            "data": {
                "first_model": first_logs,
                "second_model": second_logs
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error comparing models: {str(e)}"
        }), 500

@app.post("/delete")
@token_required
def api_delete_model(current_user):
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Invalid or missing JSON data"}), 400
            
        model_name = data.get("model_name")
        if not model_name or not isinstance(model_name, str):
            return jsonify({"status": "error", "message": "Valid model name is required"}), 400
        
        # First find the model in the database
        user_models = db.get_user_models(current_user.id)
        if not user_models:
            return jsonify({
                "status": "error",
                "message": "No models found for user"
            }), 404
            
        model_to_delete = next((model for model in user_models if model.name == model_name), None)
        
        if not model_to_delete:
            return jsonify({
                "status": "error",
                "message": "Model not found or you don't have permission to delete it"
            }), 404
        
        # Delete the model file from user-specific directory
        user_models_dir = f"trained_models/user_{current_user.id}"
        delete_model_func(model_name, model_path=f"{user_models_dir}/{model_name}.zip")
        
        # The database record will be automatically deleted due to cascade delete
        return jsonify({"status": "deleted", "message": "Model and its records deleted successfully"})
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error deleting model: {str(e)}"
        }), 500

@app.get("/test")
@token_required
def api_test(current_user):
    model_name = request.args.get("model")
    task_name = request.args.get("task")
    episodes = request.args.get("episodes", "1")

    if not model_name or not task_name:
        return "Missing parameters", 400

    # Get user-specific model path
    user_models_dir = f"trained_models/user_{current_user.id}"

    def event_stream():
        yield f"data: 🔧 Starting test for model={model_name}, task={task_name}, episodes={episodes}\n\n"
        model = mm.load_model(model_name, model_path=f"{user_models_dir}/{model_name}.zip")
        yield f"data: ✅ Loaded model\n\n"
        try:
            yield from Test_Model(model, episodes=int(episodes), task_name="pick_and_place")
        except Exception as e:
            yield f"data: ❌ test_model raised exception: {str(e)}\n\n"
        yield "event: end\ndata: done\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(port=5000)
