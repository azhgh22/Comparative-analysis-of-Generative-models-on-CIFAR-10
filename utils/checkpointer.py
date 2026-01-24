import os
import torch

class Checkpointer:
    def __init__(self, checkpoint_dir: str, model_name: str, save_every: int, del_prev: bool=True) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.save_every = save_every
        self.del_prev = del_prev

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, epoch: int, state_dict):
        # Only save every N epochs
        if epoch % self.save_every != 0:
            return

        filename = f"{self.model_name}_epoch_{epoch}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state_dict, path)

        # Delete previous checkpoints if enabled
        if self.del_prev:
            for fl in os.listdir(self.checkpoint_dir):
                full_path = os.path.join(self.checkpoint_dir, fl)
                if os.path.isfile(full_path) and fl != filename:
                    os.remove(full_path)

        print(f"Checkpoint saved: {path}")

    # ---- Helpers ---- #

    @staticmethod
    def _extract_epoch(file_name):
        """Extract epoch number from 'model_epoch_XX.pt'."""
        try:
            return int(file_name.replace(".pt", "").split("_")[-1])
        except:
            return -1

    def _get_all_checkpoints(self):
        """Return sorted list of checkpoint filenames."""
        files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self.model_name) and f.endswith(".pt")
        ]
        return sorted(files, key=self._extract_epoch)

    # ---- Loader ---- #

    def load(self, epoch_num=-1):
        files = self._get_all_checkpoints()

        if not files:
            print("No checkpoints found.")
            return None  # (state_dict)

        # Load latest
        if epoch_num == -1:
            latest = files[-1]
            path = os.path.join(self.checkpoint_dir, latest)
            checkpoint = torch.load(path, map_location="cpu")
            print(f"Loaded latest checkpoint: {path}")
            return checkpoint

        # Load specific epoch
        target_filename = f"{self.model_name}_epoch_{epoch_num}.pt"
        target_path = os.path.join(self.checkpoint_dir, target_filename)

        if os.path.exists(target_path):
            checkpoint = torch.load(target_path, map_location="cpu")
            print(f"Loaded checkpoint for epoch {epoch_num}: {target_path}")
            return checkpoint

        # If file not found
        print(f"Checkpoint for epoch {epoch_num} not found.")
        return None