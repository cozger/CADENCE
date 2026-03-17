"""Load XDF files from YQP LabRecorder sessions and extract relevant streams."""

import numpy as np


# Stream name patterns for matching
EEG_TYPE = 'EEG'
ECG_NAMES = {'P1_ecg', 'P2_ecg'}
LANDMARK_NAMES = {'P1_landmarks', 'P2_landmarks'}
POSE_NAMES = {'P1_pose', 'P2_pose'}
MARKER_TYPE = 'Markers'


def load_session(xdf_path, p1_eeg_index=0, p2_eeg_index=1):
    """
    Load a YQP session from XDF file.

    Parameters:
        xdf_path: Path to .xdf file
        p1_eeg_index: Which EmotivDataStream-EEG is P1 (0 or 1)
        p2_eeg_index: Which EmotivDataStream-EEG is P2 (0 or 1)

    Returns:
        dict with raw numpy arrays and timestamps per stream
    """
    import pyxdf
    streams, header = pyxdf.load_xdf(xdf_path)

    session = {}
    eeg_streams = []

    for s in streams:
        name = s['info']['name'][0]
        stype = s['info']['type'][0]
        timestamps = np.array(s['time_stamps'], dtype=np.float64)

        # Marker streams contain strings, not numeric data
        if stype == MARKER_TYPE:
            marker_data = s['time_series']
            if len(timestamps) > 0:
                session['markers'] = list(zip(
                    timestamps.tolist(),
                    [row[0] if isinstance(row, list) else str(row) for row in marker_data]
                ))
            continue

        try:
            data = np.array(s['time_series'], dtype=np.float64)
        except (ValueError, TypeError):
            continue

        if len(data) == 0:
            continue

        if stype == EEG_TYPE:
            eeg_streams.append((name, data, timestamps))
        elif name == 'P1_ecg':
            session['p1_ecg_raw'] = data
            session['p1_ecg_ts'] = timestamps
        elif name == 'P2_ecg':
            session['p2_ecg_raw'] = data
            session['p2_ecg_ts'] = timestamps
        elif name == 'P1_landmarks':
            session['p1_landmarks_raw'] = data
            session['p1_landmarks_ts'] = timestamps
        elif name == 'P2_landmarks':
            session['p2_landmarks_raw'] = data
            session['p2_landmarks_ts'] = timestamps
        elif name == 'P1_pose':
            session['p1_pose_raw'] = data
            session['p1_pose_ts'] = timestamps
        elif name == 'P2_pose':
            session['p2_pose_raw'] = data
            session['p2_pose_ts'] = timestamps

    # Assign EEG streams to participants
    if len(eeg_streams) >= 2:
        session['p1_eeg_raw'] = eeg_streams[p1_eeg_index][1]
        session['p1_eeg_ts'] = eeg_streams[p1_eeg_index][2]
        session['p2_eeg_raw'] = eeg_streams[p2_eeg_index][1]
        session['p2_eeg_ts'] = eeg_streams[p2_eeg_index][2]
    elif len(eeg_streams) == 1:
        session['p1_eeg_raw'] = eeg_streams[0][1]
        session['p1_eeg_ts'] = eeg_streams[0][2]

    # Detect participant roles (therapist=Sway vs patient=Y_XX)
    roles = _detect_roles(streams)
    session.update(roles)

    return session


def _detect_roles(streams):
    """
    Detect therapist ('Sway') vs patient roles from XDF stream metadata.

    Searches stream names, source_ids, and description fields for 'Sway'.
    Whichever participant owns the Sway-tagged stream is the therapist.

    Returns:
        dict with p1_role, p2_role, p1_name, p2_name (or defaults if undetected)
    """
    sway_participant = None

    for s in streams:
        name = s['info']['name'][0]
        source_id = s['info'].get('source_id', [''])[0] if 'source_id' in s['info'] else ''

        # Collect description text
        desc = ''
        if 'desc' in s['info']:
            desc_data = s['info']['desc']
            if isinstance(desc_data, list) and desc_data:
                desc = str(desc_data[0])
            elif isinstance(desc_data, str):
                desc = desc_data

        all_text = f"{name} {source_id} {desc}".lower()

        if 'sway' in all_text:
            if 'p1' in name.lower():
                sway_participant = 'p1'
            elif 'p2' in name.lower():
                sway_participant = 'p2'
            break

    if sway_participant == 'p1':
        return {
            'p1_role': 'therapist', 'p2_role': 'patient',
            'p1_name': 'Sway', 'p2_name': 'patient',
        }
    elif sway_participant == 'p2':
        return {
            'p1_role': 'patient', 'p2_role': 'therapist',
            'p1_name': 'patient', 'p2_name': 'Sway',
        }
    else:
        return {
            'p1_role': 'therapist', 'p2_role': 'patient',
            'p1_name': 'unknown', 'p2_name': 'unknown',
        }


def print_session_summary(session):
    """Print a summary of loaded session data."""
    modalities = ['eeg', 'ecg', 'landmarks', 'pose']
    participants = ['p1', 'p2']

    print("=" * 60)
    print("Session Summary")
    print("=" * 60)

    for p in participants:
        print(f"\n{p.upper()}:")
        for mod in modalities:
            raw_key = f'{p}_{mod}_raw'
            ts_key = f'{p}_{mod}_ts'
            if raw_key in session:
                data = session[raw_key]
                ts = session[ts_key]
                duration = ts[-1] - ts[0]
                srate = len(ts) / duration if duration > 0 else 0
                print(f"  {mod:12s}: {data.shape} | {duration:.1f}s | ~{srate:.1f} Hz")
            else:
                print(f"  {mod:12s}: NOT FOUND")

    if 'markers' in session:
        print(f"\nMarkers: {len(session['markers'])} events")

    if 'duration' in session:
        print(f"\nAligned duration: {session['duration']:.1f}s ({session['duration']/60:.1f} min)")
