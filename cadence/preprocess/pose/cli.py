from cadence.preprocess._cli import make_modality_cli
from cadence.preprocess.pose.pipeline import preprocess_pose_session

if __name__ == "__main__":
    make_modality_cli(preprocess_pose_session, modality_name="pose")
