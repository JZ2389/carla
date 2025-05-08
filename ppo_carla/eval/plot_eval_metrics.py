import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path="eval_metrics.csv"):
    df = pd.read_csv(csv_path, names=[
        "Town", "WeatherID", "Steps", "Reward", "Collisions", "Duration"
    ])

    plt.figure()
    df.groupby("Town")["Reward"].mean().plot(kind="bar", title="Average Reward per Town")
    plt.ylabel("Average Reward")
    plt.xlabel("Town")
    plt.tight_layout()
    plt.savefig("reward_per_town.png")

    plt.figure()
    df.groupby("WeatherID")["Collisions"].mean().plot(kind="bar", title="Avg Collisions per Weather")
    plt.ylabel("Avg Collisions")
    plt.xlabel("Weather ID")
    plt.tight_layout()
    plt.savefig("collisions_per_weather.png")

    # Steps vs Reward
    plt.figure()
    plt.scatter(df["Steps"], df["Reward"], c='blue', label='Episodes')
    plt.title("Steps vs. Reward")
    plt.xlabel("Steps")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("steps_vs_reward.png")

    print("Plots saved: reward_per_town.png, collisions_per_weather.png, steps_vs_reward.png")

if __name__ == "__main__":
    plot_metrics()
