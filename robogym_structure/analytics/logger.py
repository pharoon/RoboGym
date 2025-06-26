import matplotlib.pyplot as plt
import pandas as pd
from database.models import fetch_logs
from datetime import datetime

def get_log_dataframe(model_name: str, user_id: int) -> pd.DataFrame:
    """
    Fetches training logs and returns them as a pandas DataFrame.
    Converts the new nested progress data structure into a flat DataFrame.
    """
    logs = fetch_logs(model_name, user_id)
    data = []
    
    for session in logs:
        session_id = session['session_id']
        # Extract progress data points
        for progress in session['progress']:
            data.append({
                "session_id": session_id,
                "timestep": progress['timestep'],
                "mean_reward": progress['mean_reward'],
                "logged_at": datetime.fromisoformat(progress['logged_at']) if progress['logged_at'] else None,
                "model_name": model_name,
                "user_id": user_id
            })
    
    return pd.DataFrame(data)

def plot_model_rewards(model_name: str, user_id: int):
    """
    Plots reward over time for a specific model.
    Shows both timestep-based and time-based plots.
    """
    df = get_log_dataframe(model_name, user_id)
    if df.empty:
        print(f"No data found for model: {model_name}")
        return

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot rewards vs timesteps
    df.sort_values("timestep", inplace=True)
    ax1.plot(df["timestep"], df["mean_reward"], marker=".", linestyle="-")
    ax1.set_title(f"Training Rewards vs Timesteps: {model_name}")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Mean Reward")
    ax1.grid(True)

    # Plot rewards vs real time
    df.sort_values("logged_at", inplace=True)
    ax2.plot(df["logged_at"], df["mean_reward"], marker=".", linestyle="-")
    ax2.set_title(f"Training Rewards vs Time: {model_name}")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Mean Reward")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def compare_models(model_names: list, user_id: int):
    """
    Compares training rewards across multiple models.
    Shows both timestep-based and time-based comparisons.
    All models must belong to the same user.
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for name in model_names:
        df = get_log_dataframe(name, user_id)
        if not df.empty:
            # Plot rewards vs timesteps
            df.sort_values("timestep", inplace=True)
            ax1.plot(df["timestep"], df["mean_reward"], label=name, marker=".", linestyle="-")
            
            # Plot rewards vs real time
            df.sort_values("logged_at", inplace=True)
            ax2.plot(df["logged_at"], df["mean_reward"], label=name, marker=".", linestyle="-")

    ax1.set_title("Reward Comparison Across Models (by Timesteps)")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Mean Reward")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Reward Comparison Across Models (by Time)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Mean Reward")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_training_session(model_name: str, user_id: int, session_id: int = None):
    """
    Plots reward progression for a specific training session.
    If session_id is not provided, plots the latest session.
    """
    df = get_log_dataframe(model_name, user_id)
    if df.empty:
        print(f"No data found for model: {model_name}")
        return

    if session_id is None:
        # Get the latest session
        session_id = df["session_id"].max()

    session_df = df[df["session_id"] == session_id]
    if session_df.empty:
        print(f"No data found for session {session_id}")
        return

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot rewards vs timesteps
    session_df.sort_values("timestep", inplace=True)
    ax1.plot(session_df["timestep"], session_df["mean_reward"], marker=".", linestyle="-")
    ax1.set_title(f"Training Session {session_id} Rewards vs Timesteps: {model_name}")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Mean Reward")
    ax1.grid(True)

    # Plot rewards vs real time
    session_df.sort_values("logged_at", inplace=True)
    ax2.plot(session_df["logged_at"], session_df["mean_reward"], marker=".", linestyle="-")
    ax2.set_title(f"Training Session {session_id} Rewards vs Time: {model_name}")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Mean Reward")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
