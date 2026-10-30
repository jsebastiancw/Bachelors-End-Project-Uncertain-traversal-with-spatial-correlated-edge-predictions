import os
import torch
import torch.optim as optim
import torch.nn as nn
from Env_724 import Env
from GAT_zzt_725 import GAT


def main(device: torch.device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[MARVEL] Using compute device: {device}")
    if device.type == 'cuda':
        print(f"[MARVEL] GPU Name: {torch.cuda.get_device_name(0)}")

    data_name = 'Simple'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', f'marvel_{data_name.lower()}_gat.pth')
    emb_size = 128

    robots = [
        {
            'robot_num': 1,
            'importance': 1,
            'max_time': 4.8,
            'time_spent': 0.0,
            'start_node': 9,
            'goal_node': 12,
            'current_node': 9,
            'next_node': None,
        },
        {
            'robot_num': 2,
            'importance': 1,
            'max_time': 8.9,
            'time_spent': 0.0,
            'start_node': 1,
            'goal_node': 8,
            'current_node': 1,
            'next_node': None,
        },
    ]
    for robot in robots:
        robot['time_spent'] = torch.tensor(robot['time_spent'], dtype=torch.float32, device=device)

    env = Env(data_name, model_path, emb_size, robots, device)

    in_channels = emb_size + 40
    hidden_channels = 2 * in_channels
    out_channels = 1
    learning_rate = 1e-4

    model = GAT(
        data_name=data_name,
        learning_rate=learning_rate,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=6,
        use_wandb=False,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
    criterion = nn.CrossEntropyLoss().to(device)

    model.train_model(
        env=env,
        optimizer=optimizer,
        criterion=criterion,
        num_episodes=1000,
        device=device,
        scheduler=scheduler,
        model_path=model_path,
        emb_size=emb_size,
    )


if __name__ == "__main__":
    compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(device=compute_device)
