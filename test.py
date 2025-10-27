# delete unnecessary files

def delete_unnecessary_files():
    import os


    folder_paths = ["saved_models/model6/checkpoints/", "saved_models/model6/training_progress/"]
    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            if filename.startswith("checkpoint_step_") and filename.endswith(".pth"):
                step_number = int(filename[len("checkpoint_step_"):-len(".pth")])
                should_delete = (step_number % 10000 != 0)
            elif filename.startswith("training_progress_step_") and filename.endswith(".png"):
                step_number = int(filename[len("training_progress_step_"):-len(".png")])
                should_delete = (step_number % 10000 != 0)
            else:
                # skip files that don't match expected patterns/suffixes
                continue
            if should_delete:
                file_path = os.path.join(folder_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                except FileNotFoundError:
                    print(f"File not found, skipping: {file_path}")
                except PermissionError:
                    print(f"Permission denied, skipping: {file_path}")


delete_unnecessary_files()