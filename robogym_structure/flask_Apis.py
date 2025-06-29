# flask_api.py
from flask import Flask, json, request, jsonify, Response, stream_with_context
from flask_cors import CORS, cross_origin
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
from FileStorage import upload_to_storage, delete_from_storage,download_from_storage
# Load environment variables from Credentials.env
load_dotenv('Credentials.env')

app = Flask(__name__)
# Configure CORS to allow all origins for development
CORS(app, origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"], supports_credentials=True)

# Configure JWT
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')  # Change in production
app.config['JWT_EXPIRATION_DELTA'] = timedelta(hours=int(os.getenv('JWT_EXPIRATION_HOURS', '24')))

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
    
    if db.get_user_by_email(data["email"]):
        return jsonify({"message" : "Email already exists!"}), 400

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
    
    # token = jwt.encode({
    #     'user_id': user.id,
    #     'username': user.username,
    #     'exp': datetime.utcnow() + app.config['JWT_EXPIRATION_DELTA']
    # }, app.config['SECRET_KEY'], algorithm="HS256")
    
    return jsonify({
        # 'token': token,
        'user_id': user.id,
        'username': user.username,
        'email': user.email
    })

@app.post("/initialize")
# @token_required
def api_initialize():
    initialize()
    return jsonify({"status": "initialized"})

@app.get("/train")
# @token_required
def api_train():
    model_name = request.args.get("model_name")
    timesteps = request.args.get("timesteps")
    task_number = request.args.get("task_number")
    currUserID = request.args.get('curr_user_id')
    if not model_name or not timesteps or not task_number:
        return "Missing parameters", 400

    try:
        timesteps = int(timesteps)
        task_number = int(task_number)
    except ValueError:
        return "Invalid parameter types", 400

    # Create user-specific directories if they don't exist
    user_models_dir = f"trained_models/user_{currUserID}"
    # user_logs_dir = f"logs/user_{currUserID}"
    os.makedirs(user_models_dir, exist_ok=True)
    # os.makedirs(user_logs_dir, exist_ok=True)

    local_model_path = f"{user_models_dir}/{model_name}.zip"
    # Create a trained model record in the database
    
    
    current_timestep = 0

    def event_stream():
        nonlocal current_timestep
        yield "data: 🟢 Training started...\n\n"
        start_time = time.time()
        mean_reward = None
        logs = []

        try:
            for event in train_model_func(
                model_name=model_name,
                timesteps=timesteps,
                task_number=task_number,
                model_path=local_model_path
            ):
                # Remove 'data: ' prefix and process log lines
                line = event.strip().removeprefix("data: ").strip()

                if line.startswith("REWARD_LOG::"):
                    try:
                        _, logged_model_name, reward_str, timestep_str = line.split("::")
                        mean_reward = float(reward_str)
                        current_timestep = int(timestep_str)
                        logs.append({
                            "timestep": current_timestep,
                            "mean_reward": mean_reward
                        })
                        print("Logs are ", logs, flush=True)
                        # db.log_training(
                        #     model_name=logged_model_name,
                        #     mean_reward=mean_reward,
                        #     current_timestep=current_timestep,
                        #     user_id=currUserID
                        # )
                    except (ValueError, IndexError):
                        pass  # Ignore malformed reward logs

                yield event  # Pass the full SSE line back to the frontend
            # Save the final training session
            total_time = time.time() - start_time
            remote_model_path = f"user_{currUserID}/{model_name}.zip"
            try:
                model_url = upload_to_storage("models", local_model_path, remote_model_path)
                trained_model = db.create_trained_model(
                    name=model_name,
                    model_path=model_url,
                    algorithm=db.AlgorithmType.PPO,  # Default to PPO for now
                    robotic_arm=db.RoboticArmType.KUKA_IIWA,  # Default to KUKA_IIWA for now
                    user_id=currUserID,
                    timesteps=timesteps,
                    total_time=total_time,
                    mean_reward=mean_reward,
                )
            except Exception as e:
                return jsonify({'message': f'Error creating model record: {str(e)}'}), 500

            try:
                db.create_train_session(
                    model_id=trained_model.id if trained_model else None,
                    user_id=currUserID,
                    timesteps=timesteps,
                    total_time=total_time,
                    mean_reward=mean_reward,
                    train_log=json.dumps(logs)
                )
            except Exception as e:
                return jsonify({'message': f'Error creating training session: {str(e)}'}), 500
            

           

            # Use The Logs to update the database
            print("Final logs:", logs, flush=True)

        except Exception as e:
            yield f"data: ❌ Error: {str(e)}\n\n"
            yield "event: end\ndata: failed\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.post("/continue_train")
# @token_required
def api_continue_train():
    data = request.get_json()

    model_id = data.get("model_id")
    model_name = data.get("model_name")
    timesteps = data.get("timesteps")
    task_number = data.get("task_number")
    currUserID = data.get('curr_user_id')

    if not model_name or not timesteps or not task_number or not currUserID:
        return jsonify({"status": "error", "message": "Missing required parameters"}), 400
    try:
        timesteps = int(timesteps)
        task_number = int(task_number)
        currUserID = int(currUserID)
    except ValueError:
        return "Invalid parameter types", 400

    user_models_dir = f"trained_models/user_{currUserID}"
    os.makedirs(user_models_dir, exist_ok=True)
    local_model_path = f"{user_models_dir}/{model_name}.zip"
    remote_model_path = f"user_{currUserID}/{model_name}.zip"

    success, msg = download_from_storage("models", remote_model_path, local_model_path)
    if not success:
        return jsonify({"status": "error", "message": f"Failed to download model: {msg}"}), 500

    current_timestep = 0

    def event_stream():
        nonlocal current_timestep
        yield "data: 🔁 Continuing training...\n\n"
        start_time = time.time()
        logs = []
        mean_reward = None

        try:
            for event in train_model_func(
                model_name=model_name,
                timesteps=timesteps,
                task_number=task_number,
                model_path=local_model_path
            ):
                line = event.strip().removeprefix("data: ").strip()

                if line.startswith("REWARD_LOG::"):
                    try:
                        _, _, reward_str, timestep_str = line.split("::")
                        mean_reward = float(reward_str)
                        current_timestep = int(timestep_str)
                        logs.append({
                            "timestep": current_timestep,
                            "mean_reward": mean_reward
                        })
                    except (ValueError, IndexError):
                        pass

                yield event

            total_time = time.time() - start_time

            # Upload updated model to Supabase
            upload_to_storage("models", local_model_path, remote_model_path)
            db.update_trained_model(
                model_id=model_id,
                timesteps=timesteps,
                total_time=total_time,
                mean_reward=mean_reward if mean_reward is not None else 0.0,
                new_model_path=remote_model_path
            )
            db.create_train_session(
                model_id=model_id,
                user_id=currUserID,
                timesteps=timesteps,
                total_time=total_time,
                mean_reward=mean_reward,
                train_log=json.dumps(logs)
            )

        except Exception as e:
            yield f"data: ❌ Error: {str(e)}\n\n"
            yield "event: end\ndata: failed\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.post("/models")
# @token_required
def api_list_models():
    # Get models for the current user from the database
    data = request.json
    if not data or not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Invalid or missing JSON data"}), 400
    currUserID = data["currUserID"]
    # print("CurrUserID inside models is now ", currUserID, flush= True)
    user_models = db.get_user_models(currUserID)
    
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
                'model_path': model.model_path,
                'updated_at': model.updated_at.isoformat() if model.updated_at is not None else None,
                'total_training_time': model.total_training_time,
                'final_mean_reward': model.final_mean_reward,
                'total_timesteps': model.total_timesteps,

            }
            models_list.append(model_data)
        except AttributeError:
            # Skip models with missing attributes
            continue
    
    return jsonify(models_list)

@app.post("/upload")
# @token_required
def api_upload_model():
    data = request.json
    if not data or not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Invalid or missing JSON data"}), 400
    filePath = data.get("FilePath")
    modelName = data.get("ModelName")
    currUserID = data.get("currUserID")

    print("Data is ", data, flush=True)
    
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
        
    if not filePath or not modelName:
        return jsonify({"status": "error", "message": "Missing FilePath or ModelName"}), 400

    try:
        remote_model_path = f"user_{currUserID}/{modelName}.zip"
        model_url = upload_to_storage("models", filePath, remote_model_path)

        # Create user-specific directory if it doesn't exist
        user_models_dir = f"trained_models/user_{currUserID}"
        os.makedirs(user_models_dir, exist_ok=True)
        
        # First upload the model file to user-specific directory
        target_path = f"{user_models_dir}/{modelName}.zip"
        upload_model(file_path=filePath, model_name=modelName, target_path=target_path)
        
        # Create database record for the uploaded model
        trained_model = db.create_trained_model(
            name=modelName,
            model_path=model_url,
            algorithm=db.AlgorithmType.PPO,  # Default to PPO for uploaded models
            robotic_arm=db.RoboticArmType.KUKA_IIWA,  # Default to KUKA_IIWA for uploaded models
            user_id=currUserID
        )

       
        
        return jsonify({
            "status": "ok",
            "model_id": trained_model.id if trained_model else None,
            "message": "Model uploaded and saved to database successfully"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error uploading model: {str(e)}"
        }), 500

@app.get("/download")
# @token_required
def api_download_model():
    model_name = request.args.get("model_name")
    local_model_path = request.args.get("local_model_path")
    currUserID = request.args.get('curr_user_id')

    if not model_name or not local_model_path or not currUserID:
        return jsonify({"status": "error", "message": "Missing required parameters"}), 400
    try:
        currUserID = int(currUserID)
    except ValueError:
        return "Invalid parameter types", 400

    remote_model_path = f"user_{currUserID}/{model_name}.zip"
    success, msg = download_from_storage("models", remote_model_path, local_model_path)

    if not success:
        return jsonify({"status": "error", "message": f"Failed to download model: {msg}"}), 500
    return jsonify({"status": "ok", "message": "Model downloaded successfully"}), 200

@app.post("/getRewards")
# @token_required
def api_get_rewards():
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Invalid or missing JSON data"}), 400
        model_name = data["Model_Name"]
        currUserID = data["currUserID"]
        
        # Get the model's training history from our database
        logs = db.fetch_logs(model_name, currUserID)
        if not logs:
            return jsonify({"status": "error", "message": "No training data found for this model"}), 404

        # Start the visualization process with user-specific paths
        user_models_dir = f"trained_models/user_{currUserID}"
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
# @token_required
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
# @token_required
def api_delete_model():
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Invalid or missing JSON data"}), 400
        currUserID = data["currUserID"]
        model_name = data["model_name"]
            
        try:
            # Check if the model exists in the database for the current user
            trained_model = db.get_trained_model_by_name_and_user(currUserID, model_name)
            if not trained_model:
                return jsonify({"status": "error", "message": "Model not found"}), 404
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error fetching model: {str(e)}"}), 500
        
        
        if not (db.delete_trained_model(currUserID, model_name)):
            return jsonify({"status": "error", "message": "Failed to delete model from database"}), 500

        cloud_deleted = delete_from_storage("models", f"user_{currUserID}/{model_name}.zip")

        if not cloud_deleted:
            return jsonify({
                "status": "error",
                "message": "Failed to delete model from cloud storage"
            }), 500
        # Delete the model file from user-specific directory
        user_models_dir = f"trained_models/user_{currUserID}"
        delete_model_func(model_name, model_path=f"{user_models_dir}/{model_name}.zip")
        
        # The database record will be automatically deleted due to cascade delete
        return jsonify({"status": "deleted", "message": "Model and its records deleted successfully"})
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error deleting model: {str(e)}"
        }), 500

@app.get("/test")
# @token_required
def api_test():
    model_name = request.args.get("model")
    task_name = request.args.get("task")
    episodes = request.args.get("episodes", "1")
    currUserID = request.args.get("currUserID")

    if not model_name or not task_name:
        return "Missing parameters", 400

    # Get user-specific model path
    user_models_dir = f"trained_models/user_{currUserID}"

    def event_stream():
        yield f"data: 🔧 Starting test for model={model_name}, task={task_name}, episodes={episodes}\n\n"
        model = mm.load_model(model_name, model_path=f"{user_models_dir}/{model_name}.zip")
        yield f"data: ✅ Loaded model\n\n"
        try:
            yield from Test_Model(model, episodes=int(episodes), task_name="pick_and_place")
        except Exception as e:
            yield f"data: ❌ test_model raised exception: {str(e)}\n\n"
        yield "event: end\ndata: done\n\n"
  
        db.increment_tests_run(int(currUserID) if currUserID is not None else 0)
    
        # Start the test in a separate process
        Process(target=Test_Model, args=(model_name, task_name, int(episodes), user_models_dir)).start()
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.post("/rename")
# @token_required
def api_rename_model():
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Invalid or missing JSON data"}), 400
    model_name = data["model_name"]
    new_name = data["new_name"]
    currUserID = data["currUserID"]

    if not data or not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Invalid or missing JSON data"}), 400
    
    if not model_name or not isinstance(model_name, str):
        return jsonify({"status": "error", "message": "Valid model name is required"}), 400             
    if not new_name or not isinstance(new_name, str):
        return jsonify({"status": "error", "message": "Valid new name is required"}), 400
    
    if db.model_rename(currUserID, model_name, new_name):
        return jsonify({"status": "ok", "message": "Model renamed successfully"}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to rename model"}), 500   

@app.post("/getModelSessions")
# @token_required
def api_get_model_sessions():
    # print("We are here", flush=True)
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Invalid or missing JSON data"}), 400

    model_id = data.get("modelID")
    currUserID = data.get("currUserID")
    
    if not model_id or not isinstance(model_id, int):
        return jsonify({"status": "error", "message": "Valid model name is required"}), 400
    
    if not currUserID or not isinstance(currUserID, int):
        return jsonify({"status": "error", "message": "Valid user ID is required"}), 400
    sessions = db.get_model_sessions(model_id, currUserID)
    def session_to_dict(session):
        return {
            "id": session.id,
            "model_id": session.model_id,
            "timesteps": session.timesteps,
            "total_time": session.total_time,
            "mean_reward": session.mean_reward,
            "started_at": session.started_at,
            "completed_at": session.completed_at,
            'train_log': session.train_log,  # Assuming train_log is a JSON string
            # Add any other fields you need
        }

    sessions_list = [session_to_dict(s) for s in sessions]
    return jsonify({"status": "ok", "message": "Model sessions fetched successfully", "sessions": sessions_list}), 200

@app.post("/getUserStats")
# @token_required
def api_get_user_stats():
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Invalid or missing JSON data"}), 400
        
        currUserID = data.get("currUserID")
        if not currUserID:
            return jsonify({"status": "error", "message": "currUserID is required"}), 400
        
        try:
            currUserID = int(currUserID)
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "currUserID must be a valid integer"}), 400
        
        print("CurrUserID is ", currUserID, flush=True)
        
        stats = db.get_user_stats(currUserID)
        if not stats:
            return jsonify({"status": "error", "message": "User stats not found"}), 404

        return jsonify({"status": "ok", "message": "User stats fetched successfully", "stats": stats}), 200
    except Exception as e:
        print(f"Error in getUserStats: {str(e)}", flush=True)
        return jsonify({"status": "error", "message": f"Internal server error: {str(e)}"}), 500

@app.post("/test1")
# @token_required
def api_test1():

    if db.create_trained_model(
                    name="sa",
                    model_path="asd",
                    algorithm=db.AlgorithmType.PPO,  # Default to PPO for now
                    robotic_arm=db.RoboticArmType.KUKA_IIWA,  # Default to KUKA_IIWA for now
                    user_id=2,
                    timesteps=20,
                    total_time=20.2,
                    mean_reward=-5,
                ):
        return jsonify({"status": "ok", "message": "Model exists"}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to rename model"}), 500   


if __name__ == "__main__":
    app.run(port=5000)
