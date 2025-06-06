import os
import argparse

from model_manager import manager as mm
from rl_agent.train_agent import train_model, test_model
from database import models as db
from analytics import logger

TASK_CHOICES = {
    1: "pick_and_place",
    # 2: "button_pressing",
    # 3: "path_following",
}
def initialize():
    print("[] Initializing RoboGym environment...")
    db.init_db()
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("[] Directories and database initialized.")

def train():
    model_name = input("Enter model name to train: ")
    timesteps = int(input("Enter total training timesteps: "))

    print("\nAvailable Tasks:")
    for number, task in TASK_CHOICES.items():
        print(f"{number}. {task.replace('_', ' ').title()}")

    while True:
        try:
            choice = int(input("Choose task number: "))
            if choice in TASK_CHOICES:
                task_name = TASK_CHOICES[choice]
                break
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

    path = train_model(timesteps, model_name, task_name)
    print(f"[] Training complete. Final model saved at: {path}")

def test():
    models = mm.list_models()
    if not models:
        print("❌ No models available.")
        return

    print("\nAvailable Models:")
    for i, m in enumerate(models):
        print(f"{i + 1}. {m['name']}")

    choice = int(input("Choose a model to test: ")) - 1
    model_name = models[choice]['name']
    print("models are  now ", models)
    RL_model =mm.load_model(model_name=model_name)
    
    print("\nAvailable Tasks:")
    print("1. Pick And Place")
    task_choice = int(input("Choose task: "))
    task_name = "pick_and_place" if task_choice == 1 else None

    episodes = int(input("How many episodes to test? "))

    test_model(RL_model, task_name, episodes)


def list_models():
    models = mm.list_models()
    if not models:
        print("[!] No models found.")
    for m in models:
        print(f"• {m['name']} | Created: {m['created_at']} | Algorithm: {m['algorithm']}")

def delete():
    model_name = input("Enter model name to delete: ")
    mm.delete_model(model_name)

def analytics_menu():
    print("1. Plot rewards for a model")
    print("2. Compare multiple models")
    choice = input("Choice: ")
    if choice == "1":
        model_name = input("Model name: ")
        logger.plot_model_rewards(model_name)
    elif choice == "2":
        model_names = input("Enter model names separated by commas: ").split(",")
        model_names = [m.strip() for m in model_names]
        logger.compare_models(model_names)
    else:
        print("Invalid choice.")

def upload_model():
    from stable_baselines3 import PPO
    import shutil

    file_path = input("Enter path to existing .zip model file: ")
    model_name = input("Enter name to register the model as: ")

    if not os.path.exists(file_path):
        print("[!] File does not exist.")
        return

    dest_path = os.path.join("trained_models", f"{model_name}.zip")
    shutil.copy(file_path, dest_path)

    model = PPO.load(dest_path)
    mm.save_model(model, model_name)
    print(f"[] Model uploaded and registered as '{model_name}'")

def menu():
    print("\n==== RoboGym Control Center ====")
    print("1. Initialize system")
    print("2. Train model")
    print("3. Test model")
    print("4. List models")
    print("5. Delete model")
    print("6. View analytics")
    print("7. Upload existing model")
    print("0. Exit")

def main():
    while True:
        menu()
        choice = input("Select an option: ").strip()
        if choice == "1":
            initialize()

        elif choice == "2":
            train()
        elif choice == "3":
            test()
        elif choice == "4":
            list_models()
        elif choice == "5":
            delete()
        elif choice == "6":
            analytics_menu()
        elif choice == "7":
            upload_model()
        elif choice == "0":
            print("Goodbye.")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
