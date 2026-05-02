from cadence.preprocess._cli import make_modality_cli
from cadence.preprocess.eeg.pipeline import preprocess_eeg_session

if __name__ == "__main__":
    make_modality_cli(preprocess_eeg_session, modality_name="eeg")
