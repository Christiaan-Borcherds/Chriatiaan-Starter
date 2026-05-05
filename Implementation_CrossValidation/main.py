from kfold_pipeline import run_dev_pipeline
import config

if config.DO_DEVELOPMENT:
    output_dir = run_dev_pipeline(config)
    print(f"Development pipeline completed. Results saved to: {output_dir}")
else:
    print("DO_DEVELOPMENT=False. Nothing executed.")