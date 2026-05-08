import time
import wandb
from datetime import datetime

run = wandb.init(
    project="Starter-HAPT",
    name=f"turmite-alert-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    resume="never",
)

print("Starting W&B alert test...", flush=True)

for i in range(1, 6):
    print(f"Step {i}/5 running...", flush=True)
    wandb.log({"test_step": i})
    time.sleep(2)

wandb.alert(
    title="Turmite test finished",
    text=f"Run {run.name} completed successfully on Turmite.",
)

print("Alert sent once. Script finished.", flush=True)

wandb.finish()
