import matplotlib.pyplot as plt
import pandas as pd
from database.models import fetch_logs

def get_log_dataframe(model_name: str = None) -> pd.DataFrame:
    """
    Fetches training logs and returns them as a pandas DataFrame.
    """
    logs = fetch_logs(model_name)
    data = [{
        "model_name": log.model_name,
        "mean_reward": log.mean_reward,
        "timestamp": log.timestamp
    } for log in logs]
    
    return pd.DataFrame(data)

def plot_model_rewards(model_name: str):
    """
    Plots reward over time for a specific model.
    """
    df = get_log_dataframe(model_name)
    if df.empty:
        print(f"No data found for model: {model_name}")
        return

    df.sort_values("timestamp", inplace=True)
    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["mean_reward"], marker="o", linestyle="-")
    plt.title(f"Training Rewards Over Time: {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_models(model_names: list):
    """
    Compares training rewards across multiple models.
    """
    plt.figure(figsize=(12, 6))

    for name in model_names:
        df = get_log_dataframe(name)
        if not df.empty:
            df.sort_values("timestamp", inplace=True)
            plt.plot(df["timestamp"], df["mean_reward"], label=name)

    plt.title("Reward Comparison Across Models")
    plt.xlabel("Time")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
