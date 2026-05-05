# V8 rSLDS Ground Truth Validation Experiment

## Overview

A ~42-minute structured experiment designed to generate definitive ground truth data for validating the CADENCE V8 9-dimensional rSLDS coupling pipeline. Each block targets specific subsets of the 9D observation space with known ON/OFF coupling states, known directionality, and known temporal boundaries — enabling direct comparison between rSLDS state assignments and experimenter-controlled ground truth.

### Design Goals

1. **Temporal resolution test**: Can the constrained Viterbi (min_dwell=10s) correctly segment 10s ON/OFF blocks? How does accuracy compare with 40s blocks?
2. **Per-channel ground truth**: Every one of the 9 observation channels (EEG θ/α/β, BL expr/state, ECG LF/HF, Resp, Pose) has at least one block where it is the PRIMARY coupled channel while others are null.
3. **Directional coupling**: Leader-follower manipulation in facial mirroring validates directed coupling detection (P1→P2 vs P2→P1).
4. **Rate dissociation**: Fast vs slow breathing shifts respiratory coupling between ECG HF and LF bands — same coupling mechanism, different 9D signature.
5. **Full multimodal engagement**: End-of-session block engages all 9 channels simultaneously to test state discovery under realistic coupling.
6. **Compatible format**: Identical XDF + LSL event marker format as existing sessions.

### The 9D Observation Channels (V8 rSLDS)

| Index | Key | Source | Computation |
|-------|-----|--------|-------------|
| 0 | `eeg_theta` | fast_cycles.py | Cross-product z, 4-8 Hz band |
| 1 | `eeg_alpha` | fast_cycles.py | Cross-product z, 8-12 Hz band |
| 2 | `eeg_beta` | fast_cycles.py | Cross-product z, 13-30 Hz band |
| 3 | `bl_expr` | bl_wavelet.py | Wavelet coherence z, 0.5-2 Hz |
| 4 | `bl_state` | bl_wavelet.py | Wavelet coherence z, <0.5 Hz |
| 5 | `ecg_lf` | Hilbert envelope | Cross-product z, 0.04-0.15 Hz |
| 6 | `ecg_hf` | Hilbert envelope | Cross-product z, 0.15-0.4 Hz |
| 7 | `resp` | Phase coherence | cos(φ1 − φ2), 0.1-0.5 Hz |
| 8 | `pose` | Velocity cross-product | Upper-body velocity z, 11 joints |

### Model Constraints

- **Sampling rate**: 2 Hz (all channels resampled to common grid)
- **Constrained Viterbi**: min_dwell = 20 samples = **10 seconds**
- **Discrete states**: K = 4
- **Continuous latent**: D = 3 (SLDS dynamics)
- **Hierarchical**: Shared dynamics/transitions across sessions; session-specific emissions

---

## Equipment

Identical to existing MAP-Neuro recording setup:

| Device | Streams | LSL Type |
|--------|---------|----------|
| 2× Emotiv EPOC X | 14-ch EEG (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4) | EEG |
| 2× Polar H10 | RR intervals (130 Hz ECG) | ECG |
| 2× Face tracker | 52 blendshape coefficients @ 30 fps | Blendshapes |
| 2× Pose estimator | Body keypoints | Pose |
| 1× Marker stream | Event markers | Markers |

**Additional for this experiment:**
- Audio speaker (shared between participants) for metronome cues, breathing pace, and block transition tones
- Earpiece for participant 1 only (expression cueing in Block 3)
- Timer/stopwatch display visible to experimenter only
- Printed cards: survival item list (Block 5d), expression cue cards (Block 3 backup)

---

## Protocol

### Block 1: Null Baseline (4:00)

**Purpose**: Establish NULL reference for all 9 channels. Individual EEG signatures without inter-brain coupling.

| Sub-block | Duration | Condition | Instructions |
|-----------|----------|-----------|--------------|
| 1a | 2:00 | `null_EO` | "Sit comfortably, eyes open, look at the fixation point on your screen. Do not interact." |
| 1b | 2:00 | `null_EC` | "Close your eyes. Sit quietly and relax. Do not interact." |

**Expected 9D signature**: All channels ≈ 0. High individual alpha power during EC but zero inter-brain coupling. This is the universal null reference.

**Markers**: `null_EO_start`, `null_EO_stop`, `null_EC_start`, `null_EC_stop`

---

### Block 2: Motor Synchrony — Temporal Resolution Test (8:00)

**Purpose**: Primary test of the 10s min_dwell constraint. Does the rSLDS correctly detect rapid 10s ON/OFF alternations? Comparison with 40s blocks provides dose-response for temporal resolution. Primary target: **Pose** channel. Secondary: **EEG β** (frontal coordination, may be weak on EPOC due to limited motor cortex coverage).

**Motor action**: Synchronized bilateral arm raise/lower at 1 Hz metronome (raise on beat 1, lower on beat 2). Clearly visible in upper-body pose estimation. Audio metronome provides synchronization cue.

| Sub-block | Duration | Condition | Instructions |
|-----------|----------|-----------|--------------|
| 2a | 1:00 | `motor_solo` | "Move your arms up and down at your own pace. No need to match your partner." |
| 2b | 2:00 | `motor_rapid` | 6 cycles of: 10s sync (metronome ON) → 10s rest (metronome OFF). Audio tone marks each transition. |
| 2c | 0:40 | `motor_rest_trans` | "Rest your arms. We'll continue with longer blocks." |
| 2d | 2:40 | `motor_slow` | 2 cycles of: 40s sync (metronome ON) → 40s rest (metronome OFF). |
| 2e | 1:40 | `motor_continuous` | "Follow the metronome together continuously until I say stop." |

**Marker scheme for rapid alternation (Block 2b)**:
```
motor_rapid_sync_01_start → (10s) → motor_rapid_sync_01_stop
motor_rapid_rest_01_start → (10s) → motor_rapid_rest_01_stop
motor_rapid_sync_02_start → (10s) → motor_rapid_sync_02_stop
motor_rapid_rest_02_start → (10s) → motor_rapid_rest_02_stop
... (6 cycles = 12 condition intervals, 24 markers)
```

**Marker scheme for slow alternation (Block 2d)**:
```
motor_slow_sync_01_start → (40s) → motor_slow_sync_01_stop
motor_slow_rest_01_start → (40s) → motor_slow_rest_01_stop
motor_slow_sync_02_start → (40s) → motor_slow_sync_02_stop
motor_slow_rest_02_start → (40s) → motor_slow_rest_02_stop
```

**Other markers**: `motor_solo_start/stop`, `motor_rest_trans_start/stop`, `motor_continuous_start/stop`

**Expected 9D signature**:

| Condition | Pose | EEG β | Others |
|-----------|------|-------|--------|
| motor_solo | 0 | 0 | 0 |
| motor_*_sync | ++ | + | 0 |
| motor_*_rest | 0 | 0 | 0 |
| motor_continuous | ++ | + | 0 |

**Validation metrics**:
- **State detection accuracy**: % of 10s sync blocks correctly assigned to a motor-coupled state
- **Onset latency**: Mean time from sync_start marker to first coupled-state assignment (expect 0-5s)
- **10s vs 40s accuracy**: Detection rate for 10s blocks vs 40s blocks
- **Dose-response**: Coupling magnitude: continuous > 40s > 10s (more time = stronger coupling estimate)

---

### Block 3: Facial Expression Mirroring (6:00)

**Purpose**: Ground truth for **BL expression** and **BL state** channels. Tests directional coupling: P1→P2 in first half, P2→P1 in second half.

**Method**: Experimenter cues the leader via earpiece with expression names (smile, surprise, frown, neutral). Each expression is held for ~6-8 seconds. The follower watches the leader's face and mirrors what they see. The follower does NOT hear the cues — they mirror purely from visual observation, producing a natural 1-3s lag (mimicry latency).

**Expression sequence** (same for both directions):
1. Smile (8s) → Neutral (4s) → Surprise (6s) → Neutral (4s) → Frown (6s) → Neutral (4s) → Smile (6s) → Neutral (4s) → Surprise (8s) → Neutral (4s) → Frown (6s) → ... fills 2 min

| Sub-block | Duration | Condition | Instructions |
|-----------|----------|-----------|--------------|
| 3a | 1:00 | `face_neutral` | "Look at each other. Keep a relaxed, neutral face." |
| 3b | 2:00 | `face_P1_leads` | To P1 (earpiece): "Make the expressions I name. Hold each one clearly." To P2: "Mirror whatever you see on your partner's face." |
| 3c | 0:30 | `face_rest` | "Relax. Neutral face." |
| 3d | 2:00 | `face_P2_leads` | Roles reversed. P2 gets earpiece cues, P1 mirrors. |
| 3e | 0:30 | `face_rest_2` | "Relax." |

**Markers**: `face_neutral_start/stop`, `face_P1_leads_start/stop`, `face_rest_start/stop`, `face_P2_leads_start/stop`, `face_rest_2_start/stop`

**Optional fine-grained markers**: Individual expression onset markers (e.g., `expr_smile_start`, `expr_surprise_start`) for post-hoc AU-level validation. These are single-event markers (no stop needed).

**Expected 9D signature**:

| Condition | BL expr | BL state | EEG α | Others |
|-----------|---------|----------|-------|--------|
| face_neutral | 0 | 0 | + (gaze) | 0 |
| face_P1_leads | ++ | + | + | 0 |
| face_P2_leads | ++ | + | + | 0 |

**Validation metrics**:
- **BL expression coupling**: z-score during mirroring vs neutral (expect z > 2)
- **Directionality index**: DI = (coupling_leader→follower − coupling_follower→leader) / sum. Should be positive and flip between blocks 3b and 3d.
- **Mimicry lag**: Cross-correlation peak lag between leader and follower AU timeseries (expect 1-3s, consistent with y_06 real mimicry lag of 2.9s)

**Counterbalancing**: Alternate which participant is P1 vs P2 across dyads.

---

### Block 4: Respiratory Synchrony (6:00)

**Purpose**: Ground truth for **Resp**, **ECG HF**, and **ECG LF** channels. The fast/slow breathing rate dissociation tests whether the rSLDS can distinguish physiologically distinct states that produce coupling in different ECG frequency bands.

**Key physiological insight**: At 6 bpm (0.10 Hz), respiratory sinus arrhythmia (RSA) falls in the ECG **LF** band (0.04-0.15 Hz). At 20 bpm (0.33 Hz), RSA falls in the ECG **HF** band (0.15-0.4 Hz). Both produce respiratory coupling, but the cardiac signature appears in different frequency bands.

**Audio cues**: Breathing pace guided by a repeating audio pattern:
- **Fast (20 bpm)**: Tone on/off every 1.5s (1.5s inhale, 1.5s exhale = 3s cycle)
- **Slow (6 bpm)**: Tone sweep every 5s (5s inhale, 5s exhale = 10s cycle)
- Both participants hear the same audio cue via shared speaker.

| Sub-block | Duration | Condition | Instructions |
|-----------|----------|-----------|--------------|
| 4a | 1:00 | `breath_natural` | "Close your eyes. Breathe naturally at your own pace." |
| 4b | 2:00 | `breath_fast` | "Follow the breathing cue. Inhale when the tone rises, exhale when it falls. The pace will be brisk." |
| 4c | 1:00 | `breath_transition` | "Breathe naturally for a moment. The pace will change." |
| 4d | 2:00 | `breath_slow` | "Follow the breathing cue again. This time the pace is slow and deep." |

**Markers**: `breath_natural_start/stop`, `breath_fast_start/stop`, `breath_transition_start/stop`, `breath_slow_start/stop`

**Expected 9D signature**:

| Condition | Resp | ECG HF | ECG LF | Others |
|-----------|------|--------|--------|--------|
| breath_natural | 0 | 0 | 0 | 0 |
| breath_fast (20 bpm = 0.33 Hz) | ++ | ++ | 0 | 0 |
| breath_transition | + (declining) | + (declining) | 0 | 0 |
| breath_slow (6 bpm = 0.10 Hz) | ++ | 0 | ++ | 0 |

**Validation metrics**:
- **Resp phase coherence**: Should be high (> 0.7) during both cued conditions, near-chance during natural
- **ECG band dissociation**: Fast → ECG HF high / LF unchanged; Slow → ECG LF high / HF unchanged
- **Rate transition detection**: Does the rSLDS detect the fast→slow transition as a state change even though Resp coupling remains high? (It should, because the ECG signature changes.)

---

### Block 5: EEG Coupling Induction (10:00)

**Purpose**: Ground truth for **EEG θ/α/β** channels. Uses three literature-validated paradigms that produce progressively stronger and band-specific inter-brain synchrony (IBS).

**Literature basis**:
- **Mutual gaze**: Dikker 2017 (12 dyads, Emotiv EPOC) demonstrated that face-to-face orientation is the gate variable for alpha-band IBS. Social closeness modulates effect size. This is the most directly replicated finding with our exact hardware.
- **Cooperative problem-solving**: Meta-analytic g=1.98 (Czeszumski 2022); 13/13 fNIRS PFC studies showed significant coupling during tangrams and similar tasks. Engages alpha (joint attention) + beta (coordination) + theta (cognitive processing).
- **Emotional sharing**: Theta-specific IBS during emotional processing (Chen 2022: r=0.37). Directly models the therapeutic alliance scenario.
- **Eyes-closed, no interaction**: Gold-standard null for EEG IBS (Dikker 2017 control). High individual alpha power but zero inter-brain coupling.

| Sub-block | Duration | Condition | Instructions |
|-----------|----------|-----------|--------------|
| 5a | 2:00 | `eeg_null_EC` | "Close your eyes. Sit quietly. No interaction." Strongest EEG null — no visual, no social, no task. |
| 5b | 2:00 | `eeg_gaze` | "Open your eyes. Look into your partner's eyes silently. Just hold mutual gaze." |
| 5c | 1:00 | `eeg_rest_EC` | "Close your eyes briefly. We'll start a new task shortly." |
| 5d | 3:00 | `eeg_coop` | "Open your eyes. Here's a scenario: You're stranded on a deserted island. Here's a list of 20 items. Together, agree on the 5 most important items to survive. Discuss and decide together." |
| 5e | 2:00 | `eeg_emotion` | "Now, [P1/P2], share a personal memory that was emotionally meaningful to you — happy or sad. [Other participant], listen attentively and empathetically." |

**Markers**: `eeg_null_EC_start/stop`, `eeg_gaze_start/stop`, `eeg_rest_EC_start/stop`, `eeg_coop_start/stop`, `eeg_emotion_start/stop`

**Expected 9D signature**:

| Condition | EEG θ | EEG α | EEG β | BL expr | Others |
|-----------|-------|-------|-------|---------|--------|
| eeg_null_EC | 0 | 0 | 0 | 0 | 0 |
| eeg_gaze | 0 | ++ | 0 | + (spontaneous mimicry) | 0 |
| eeg_rest_EC | 0 | 0 | 0 | 0 | 0 |
| eeg_coop | + | ++ | + | + (conversation) | + (gesture) |
| eeg_emotion | ++ | + | 0 | + (empathic) | 0 |

**Validation metrics**:
- **Parametric ordering**: eeg_coop > eeg_gaze > eeg_null on combined EEG z (expect monotonic increase)
- **Band specificity**: Alpha dominant in gaze, theta dominant in emotion, all bands in cooperation
- **Effect size**: Compare real-dyad EEG coupling vs pseudo-dyad (cross-session pairing) per condition
- **Onset/offset**: EEG coupling should emerge within 15-30s of condition onset (literature typical)

**Cooperative task details**: Print a card with 20 survival items (e.g., knife, rope, mirror, water container, matches, compass, map, blanket, fishing line, first aid kit, tarp, flashlight, whistle, sunscreen, insect repellent, cooking pot, binoculars, axe, duct tape, chocolate). Both participants see the list. They must discuss and agree on their top 5. This task reliably produces engaged, face-to-face discussion with natural turn-taking, joint attention to the item list, and genuine cooperation.

**Emotional sharing**: Counterbalance which participant shares across dyads. The sharer speaks naturally; the listener maintains eye contact and responds naturally (nods, facial expressions) but does not interrupt.

---

### Block 6: Full Multimodal Engagement (6:00)

**Purpose**: Stress test for all 9 channels simultaneously. Tests ecological validity — does the rSLDS produce meaningful state assignments during naturalistic multi-channel coupling? Provides the closest analog to real therapy sessions.

| Sub-block | Duration | Condition | Instructions |
|-----------|----------|-----------|--------------|
| 6a | 2:00 | `multi_conv` | "Have a natural conversation about what you've been up to recently. Just chat normally." |
| 6b | 2:00 | `multi_engaged` | "Now talk about something you feel strongly about — a cause, a passion, something that excites you. Also, try to subtly match your partner's posture and movements as you talk." |
| 6c | 2:00 | `multi_winddown` | "Close your eyes. Sit quietly. Let yourself wind down." |

**Markers**: `multi_conv_start/stop`, `multi_engaged_start/stop`, `multi_winddown_start/stop`

**Expected 9D signature**:

| Condition | EEG θ | EEG α | EEG β | BL expr | BL state | ECG LF | ECG HF | Resp | Pose |
|-----------|-------|-------|-------|---------|----------|--------|--------|------|------|
| multi_conv | + | + | 0 | + | 0 | 0 | 0 | 0 | + |
| multi_engaged | ++ | ++ | + | ++ | + | + | 0 | + | ++ |
| multi_winddown | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Validation metrics**:
- **State separation**: rSLDS should assign different dominant states to conv vs engaged vs winddown
- **Channel hierarchy**: During multi_engaged, expect EEG > BL > Pose > ECG > Resp (matching y_06 findings)
- **Declining coupling**: multi_winddown should show monotonically declining coupling over 2 minutes

---

## Complete Timeline

```
00:00 ─── Block 1: Null Baseline ─────────────────────────────
00:00   1a  null_EO                 (2:00)  NULL all channels
02:00   1b  null_EC                 (2:00)  NULL all channels
04:00 ─── Block 2: Motor Temporal Resolution ──────────────────
04:00   2a  motor_solo              (1:00)  NULL (independent movement)
05:00   2b  motor_rapid             (2:00)  6× [10s sync → 10s rest]
07:00   2c  motor_rest_trans        (0:40)  Transition
07:40   2d  motor_slow              (2:40)  2× [40s sync → 40s rest]
10:20   2e  motor_continuous        (1:40)  Sustained sync
12:00 ─── Block 3: Facial Mirroring ───────────────────────────
12:00   3a  face_neutral            (1:00)  NULL BL
13:00   3b  face_P1_leads           (2:00)  P1→P2 directional BL
15:00   3c  face_rest               (0:30)  Rest
15:30   3d  face_P2_leads           (2:00)  P2→P1 directional BL
17:30   3e  face_rest_2             (0:30)  Rest
18:00 ─── Block 4: Respiratory Synchrony ──────────────────────
18:00   4a  breath_natural          (1:00)  NULL respiratory
19:00   4b  breath_fast             (2:00)  20 bpm → Resp + ECG HF
21:00   4c  breath_transition       (1:00)  Natural recovery
22:00   4d  breath_slow             (2:00)  6 bpm → Resp + ECG LF
24:00 ─── Block 5: EEG Coupling Induction ─────────────────────
24:00   5a  eeg_null_EC             (2:00)  NULL EEG (eyes closed)
26:00   5b  eeg_gaze                (2:00)  Mutual gaze → alpha IBS
28:00   5c  eeg_rest_EC             (1:00)  Brief null
29:00   5d  eeg_coop                (3:00)  Desert island task → all EEG bands
32:00   5e  eeg_emotion             (2:00)  Emotional sharing → theta IBS
34:00 ─── Block 6: Full Multimodal ────────────────────────────
34:00   6a  multi_conv              (2:00)  Natural conversation → medium all
36:00   6b  multi_engaged           (2:00)  Engaged + posture mirroring → high all
38:00   6c  multi_winddown          (2:00)  Eyes-closed wind-down → declining
40:00 ─── End ─────────────────────────────────────────────────
```

**Total recording time**: 40:00
**Total with inter-block instructions** (~30s × 5 transitions): ~42:30

---

## Ground Truth Prediction Matrix

Complete predicted 9D coupling signature for each condition. This is the reference for rSLDS validation.

```
                     EEG_θ  EEG_α  EEG_β  BL_ex  BL_st  ECG_L  ECG_H  Resp   Pose
                     ─────  ─────  ─────  ─────  ─────  ─────  ─────  ─────  ─────
null_EO                0      0      0      0      0      0      0      0      0
null_EC                0      0      0      0      0      0      0      0      0
motor_solo             0      0      0      0      0      0      0      0      0
motor_*_sync           0      0      +      0      0      0      0      0      ++
motor_*_rest           0      0      0      0      0      0      0      0      0
motor_continuous       0      0      +      0      0      0      0      0      ++
face_neutral           0      +      0      0      0      0      0      0      0
face_P1_leads          0      +      0      ++     +      0      0      0      0
face_P2_leads          0      +      0      ++     +      0      0      0      0
breath_natural         0      0      0      0      0      0      0      0      0
breath_fast            0      0      0      0      0      0      ++     ++     0
breath_slow            0      0      0      0      0      ++     0      ++     0
breath_transition      0      0      0      0      0      0      +↓     +↓     0
eeg_null_EC            0      0      0      0      0      0      0      0      0
eeg_gaze               0      ++     0      +      0      0      0      0      0
eeg_rest_EC            0      0      0      0      0      0      0      0      0
eeg_coop               +      ++     +      +      0      0      0      0      +
eeg_emotion            ++     +      0      +      +      +      0      0      0
multi_conv             +      +      0      +      0      0      0      0      +
multi_engaged          ++     ++     +      ++     +      +      0      +      ++
multi_winddown         0↓     0↓     0      0      0      0      0      0      0
```

Legend: `0` = null, `+` = moderate coupling, `++` = strong coupling, `↓` = declining

### Expected rSLDS State Mapping (K=4)

Given K=4 discrete states, the model should discover approximately:

| State | Dominant Channels | Primary Conditions |
|-------|------------------|--------------------|
| S0: **Disengaged** | All ≈ 0 | null_EO/EC, motor_rest, face_neutral, breath_natural, eeg_null, winddown |
| S1: **Motor coupling** | Pose ++, EEG β + | motor_sync, motor_continuous |
| S2: **Facial/emotional** | BL expr ++, EEG θ +, BL state + | face_P1/P2_leads, eeg_emotion |
| S3: **Full engagement** | EEG α ++, BL +, Pose +, Resp + | eeg_coop, multi_engaged, multi_conv |

The breathing blocks are interesting edge cases — they don't neatly fit the therapy-derived states. They may form a distinct cluster or split across S0 (since only ECG/Resp channels are active while all social channels are null).

---

## LSL Event Marker Specification

### Marker Format

Identical to existing sessions. Each condition interval has a `{name}_start` and `{name}_stop` marker pair sent via the LSL Markers stream.

```python
# Marker stream type (must match cadence/data/xdf_loader.py)
MARKER_TYPE = 'Markers'

# All condition names (for cadence/conditions.py integration)
GROUND_TRUTH_CONDITIONS = [
    # Block 1
    'null_EO', 'null_EC',
    # Block 2 (envelope)
    'motor_solo', 'motor_rest_trans', 'motor_continuous',
    # Block 2 (fine-grained rapid: 01-06)
    'motor_rapid_sync_01', 'motor_rapid_rest_01',
    'motor_rapid_sync_02', 'motor_rapid_rest_02',
    'motor_rapid_sync_03', 'motor_rapid_rest_03',
    'motor_rapid_sync_04', 'motor_rapid_rest_04',
    'motor_rapid_sync_05', 'motor_rapid_rest_05',
    'motor_rapid_sync_06', 'motor_rapid_rest_06',
    # Block 2 (fine-grained slow: 01-02)
    'motor_slow_sync_01', 'motor_slow_rest_01',
    'motor_slow_sync_02', 'motor_slow_rest_02',
    # Block 3
    'face_neutral', 'face_P1_leads', 'face_rest',
    'face_P2_leads', 'face_rest_2',
    # Block 4
    'breath_natural', 'breath_fast',
    'breath_transition', 'breath_slow',
    # Block 5
    'eeg_null_EC', 'eeg_gaze', 'eeg_rest_EC',
    'eeg_coop', 'eeg_emotion',
    # Block 6
    'multi_conv', 'multi_engaged', 'multi_winddown',
]
```

### Integration with Existing Pipeline

Add to `cadence/conditions.py`:

```python
# Ground truth experiment conditions
GROUND_TRUTH_DISPLAY = {
    'null_EO': 'Null (EO)',
    'null_EC': 'Null (EC)',
    'motor_solo': 'Motor Solo',
    'motor_rapid_sync': 'Motor Sync 10s',
    'motor_rapid_rest': 'Motor Rest 10s',
    'motor_slow_sync': 'Motor Sync 40s',
    'motor_slow_rest': 'Motor Rest 40s',
    'motor_rest_trans': 'Motor Transition',
    'motor_continuous': 'Motor Continuous',
    'face_neutral': 'Face Neutral',
    'face_P1_leads': 'Face P1→P2',
    'face_P2_leads': 'Face P2→P1',
    'face_rest': 'Face Rest',
    'breath_natural': 'Breath Natural',
    'breath_fast': 'Breath Fast (20bpm)',
    'breath_slow': 'Breath Slow (6bpm)',
    'breath_transition': 'Breath Transition',
    'eeg_null_EC': 'EEG Null (EC)',
    'eeg_gaze': 'Mutual Gaze',
    'eeg_rest_EC': 'EEG Rest (EC)',
    'eeg_coop': 'Cooperative Task',
    'eeg_emotion': 'Emotional Sharing',
    'multi_conv': 'Conversation',
    'multi_engaged': 'Engaged Conversation',
    'multi_winddown': 'Wind-down',
}
```

---

## Validation Analysis Plan

### 1. Temporal Resolution Validation (Block 2)

```
For each motor sync/rest block:
  1. Run rSLDS with constrained Viterbi (min_dwell=20, K=4)
  2. For each 10s sync block:
     - Compute overlap between "coupled state" and [sync_start, sync_stop] interval
     - Measure onset latency: time from sync_start to first coupled-state assignment
     - Measure offset latency: time from sync_stop to first non-coupled assignment
  3. Compare 10s accuracy vs 40s accuracy
  4. Report: detection rate, mean onset latency, mean offset latency
```

**Pass criteria**:
- 40s blocks: ≥ 90% detection rate (at least 1.8/2 correctly identified)
- 10s blocks: ≥ 60% detection rate (at least 3.6/6 correctly identified)
- Onset latency: < 5s (< 10 samples at 2 Hz)
- Coupling magnitude: motor_continuous > motor_slow_sync > motor_rapid_sync

### 2. Directional Coupling Validation (Block 3)

```
For face_P1_leads vs face_P2_leads:
  1. Compute BL expression coupling with directional lag analysis
  2. Compute directionality index: DI = (coupling_A→B - coupling_B→A) / (coupling_A→B + coupling_B→A)
  3. DI should flip sign between P1_leads and P2_leads blocks
```

**Pass criteria**:
- BL expression z > 2.0 during mirroring blocks
- DI sign correctly predicts leader in ≥ 80% of dyads
- Mimicry lag 1-4s (consistent with literature and y_06 data)

### 3. ECG Band Dissociation (Block 4)

```
For breath_fast vs breath_slow:
  1. Compute ECG LF and HF coupling z-scores per condition
  2. Test interaction: breath_fast should have HF > LF; breath_slow should have LF > HF
  3. Respiratory coupling should be high in both
```

**Pass criteria**:
- Resp coupling z > 2.0 in both cued conditions
- ECG HF significantly higher in breath_fast than breath_slow (paired t-test p < 0.05)
- ECG LF significantly higher in breath_slow than breath_fast

### 4. EEG Parametric Ordering (Block 5)

```
For eeg_null_EC < eeg_gaze < eeg_coop:
  1. Compute combined EEG coupling (mean of θ, α, β z-scores) per condition
  2. Test monotonic ordering via Page's trend test
  3. Compute band-specific contrasts: α for gaze, θ for emotion
```

**Pass criteria**:
- Combined EEG: eeg_coop > eeg_gaze > eeg_null (monotonic, p < 0.05)
- Alpha-band: eeg_gaze > eeg_null (p < 0.05)
- Theta-band: eeg_emotion > eeg_null (p < 0.05)
- Real-dyad > pseudo-dyad for eeg_coop and eeg_gaze (p < 0.05)

### 5. Full rSLDS State Validation (All Blocks)

```
Across entire 40-min session:
  1. Fit rSLDS (K=4, D_latent=3, constrained Viterbi min_dwell=20)
  2. Compute condition-state alignment: for each condition, what % of time is spent in each state?
  3. Compute mutual information between ground-truth condition labels and rSLDS state assignments
  4. Compare: MI(rSLDS, ground_truth) > MI(k-means, ground_truth)
```

**Pass criteria**:
- Null conditions (null_EO/EC, motor_rest, face_neutral, breath_natural, eeg_null_EC, winddown): ≥ 70% in disengaged state
- Motor sync conditions: ≥ 60% in a motor-coupling state distinct from facial/engagement states
- Face mirroring conditions: ≥ 60% in a facial-coupling state
- Multi_engaged: highest state diversity (entropy) — should engage multiple coupled states
- MI(rSLDS) > MI(k-means) + 0.05

---

## Counterbalancing

| Variable | Scheme |
|----------|--------|
| P1/P2 role assignment | Alternate across dyads (odd dyads: participant A = P1; even: participant A = P2) |
| Expression mirroring leader | Always P1 first, P2 second (role assignment handles counterbalancing) |
| Emotional sharing speaker | Odd dyads: P1 shares; even dyads: P2 shares |
| Survival item list | Use 2 alternate lists to prevent learning effects if re-testing |

---

## Practical Checklist

### Pre-Session (~15 min)

- [ ] Both EEG headsets fitted, impedances checked, LSL streaming confirmed
- [ ] Both Polar H10 chest straps on, ECG streaming to LSL
- [ ] Both cameras positioned, face tracker running, blendshapes streaming
- [ ] Pose estimation running for both participants
- [ ] Marker stream created in LSL
- [ ] Audio system tested (metronome, breathing cues, transition tones)
- [ ] P1 earpiece tested (for expression cueing)
- [ ] Survival item card printed and ready
- [ ] Participants seated face-to-face, ~1m apart
- [ ] Brief practice: 10s of synchronized arm movement to verify comfort
- [ ] Recording started, initial test markers sent and confirmed in XDF

### During Session

- [ ] Experimenter sends markers at each block transition
- [ ] Verbal instructions delivered at each transition (~15-30s, not counted in block time)
- [ ] Monitor EEG quality indicator — note any electrode issues
- [ ] Note any protocol deviations (e.g., participant opens eyes during EC block)

### Post-Session (~5 min)

- [ ] Stop all recordings
- [ ] Verify XDF file contains all expected marker pairs
- [ ] Quick quality check: open XDF, verify all streams present and aligned
- [ ] Participant debrief: ask about comfort, any issues

---

## Sample Size

- **Minimum**: 8 dyads (16 participants) — sufficient for within-subject temporal resolution test
- **Recommended**: 12 dyads (24 participants) — allows 4 losses, enables pseudo-dyad validation (12 real dyads → 132 pseudo-dyad combinations)
- **Demographics**: Same as MAP-Neuro recruitment (adults 25-55, right-handed, no neurological conditions)
- **Relationship**: Strangers preferred (Ohayon & Gordon 2025: strangers r=0.40 vs familiar r=0.15 — stronger coupling effects with strangers, paradoxically)

---

## Pseudo-Dyad Control

After collecting all dyads, pair P1 from dyad X with P2 from dyad Y for every possible cross-dyad combination:
- 12 dyads → 132 pseudo-pairs
- All coupling should vanish in pseudo-pairs EXCEPT:
  - Motor sync blocks (stimulus-driven via shared metronome — pseudo-pairs hear same cue) → some residual expected
  - Breathing blocks (stimulus-driven via shared audio cue) → some residual expected
  - All other blocks (interaction-driven) → should be null

This ISC vs IBC dissociation is itself a validation: stimulus-driven coupling survives pseudo-pairing; interaction-driven coupling does not.

---

## Relationship to Existing Ground Truth Doc

This experiment replaces and supersedes the design in `docs/ground_truth_paradigm.md` (63-min, 6-paradigm general validation battery). Key differences:

| Aspect | Original (63 min) | This Design (42 min) |
|--------|-------------------|---------------------|
| Primary purpose | General pipeline validation | V8 rSLDS 9D state validation |
| Duration | 63 min (too long) | 42 min (practical) |
| Temporal resolution test | Not included | Core feature (10s/40s blocks) |
| EEG paradigm specificity | Shared audio (ISC, not IBC) | Mutual gaze + cooperative task (IBC) |
| Respiratory | Guided breathing only | Fast/slow dissociation (ECG band test) |
| EDA | Required (new hardware) | Not required (existing hardware only) |
| Directionality test | Voice following (P4) | Facial mirroring (clearer BL ground truth) |
| rSLDS-specific validation | Not designed for it | Every block targets specific 9D channels |

---

## Future Extensions

1. **Add EDA**: If Shimmer3 GSR+ acquired, add a 2-min startle block (loud tones → synchronized SCR) and a 2-min emotional discussion block. EDA is the strongest therapy synchrony signal (r=0.32-0.47) but requires additional hardware.

2. **Add speech/prosody**: Lavalier microphones + OpenSMILE feature extraction. Extend Block 6 with structured turn-taking to test vocal pitch (expected negative coupling) vs linguistic style matching (expected positive).

3. **Drug-state version**: For dosing sessions, adapt Block 5 (EEG) to eyes-closed with therapist voice guidance only. Add LZc complexity as a 10th observation channel. Replace cooperative task with music listening (both hear same playlist).

4. **Longitudinal**: Run this experiment at sessions 1, 4, 8, 12 of a treatment course to track coupling trajectory (Sened 2025: d=1.34 increase across therapy sessions).
