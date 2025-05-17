
import matplotlib.pyplot as plt
import pandas as pd
from database.models import SessionLocal, TrainingLog

def plot_rewards():
    session = SessionLocal()
    logs = session.query(TrainingLog).all()
    rewards = [log.reward for log in logs]
    timestamps = [log.timestamp for log in logs]
    plt.plot(timestamps, rewards)
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Time")
    plt.show()
