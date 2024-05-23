from pathlib import Path
from trackmania_rl import run_to_video

base_dir = Path(__file__).resolve().parents[1]
tmi_dir = Path("C:\\Users\\DuFra\\OneDrive\\Dokumenter\\TMInterface\\Scripts")

# Define a variable for the folder number
folder_number = "9930"  # You can change this value as needed
sub_folder = "baseline A10"
# Update the run_dir to include the folder number
#run_dir = Path(f"E:\\Trackmania AI coding\\trackmania_rl_public\\save\\rerun_checkpoint\\best_runs\\")
run_dir = Path(f"E:\\Trackmania AI coding\\trackmania_rl_public\\save\\{sub_folder}\\best_runs\\{folder_number}")

out_dir = tmi_dir / "output for inputfiles"
out_dir.mkdir(parents=False, exist_ok=True)

# # Iterate over the subdirectories in the updated run_dir
for a in run_dir.iterdir():
    run_to_video.write_actions_from_disk_in_tmi_format(infile_path=run_dir / "actions.joblib", outfile_path=out_dir / f"{sub_folder}_{folder_number}.inputs")

# Uncomment and adjust if you need to create a widget video from q_values
# run_to_video.make_widget_video_from_q_values_on_disk(
#     q_values_path=run_dir / "133800" / "q_values.joblib", video_path=base_dir / "124910.mov", q_value_gap=0.03
# )
