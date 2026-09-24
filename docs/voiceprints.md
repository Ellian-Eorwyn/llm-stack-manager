# Voice profiles

Diarization (see `docs/mlx-macos.md`) says *that* speakers differ: `speaker_0`,
`speaker_1`, in order of arrival in one recording. Voice profiles say *who*
they are, by comparing each diarized speaker's voice with people enrolled from
transcripts someone has already labelled.

**These are biometric data about other people.** Profiles and clip embeddings
live in `models/voiceprints/`, which is git-ignored. Keep them on this machine.

## How it works

1. **Embedding.** `scripts/speaker_embedding.py` turns a few seconds of speech
   into a 256-dim vector with WeSpeaker ResNet34 (pyannote's
   `wespeaker-voxceleb-resnet34-LM`, MIT, 6.6M parameters) on MLX. Same
   speaker → nearby vectors, whatever they say.
2. **Enrollment.** `scripts/enroll-voiceprints.py macwhisper` reads MacWhisper's
   library **read-only**. Every transcript line with a real name on it is a
   clip of that person's voice at a known time. It keeps lines of at least 2 s
   that no other line overlaps, trims 0.15 s off each end, and caps them at
   12 s and 40 per person per recording (longest first). A meeting's clip
   comes from whichever of its mic and app tracks is louder at that moment,
   so a remote speaker's sample doesn't include your own voice. Generic
   labels ("Speaker 3") and MacWhisper's per-meeting "Microphone" stubs are
   skipped.
3. **Profiles.** Each person's profile is the mean direction of their clips.
   Clips that disagree with the majority (mislabelled lines, crosstalk, noise)
   are dropped first: under 0.25, or more than three robust deviations below
   the median.
4. **Identification.** With `diarize=true`, the transcription server takes each
   diarized speaker's clean stretches (up to 20, nobody else talking),
   averages their embeddings, and compares the result with every profile.
   A name is given only when the score clears the threshold and beats the
   next-best person by the margin. No name goes to two speakers.

## Using it

```bash
# Build (or rebuild after labelling more in MacWhisper). Minutes, not hours.
deps/mlx-runtime-venv/bin/python scripts/enroll-voiceprints.py macwhisper \
    --same-person "Dana=Dana (Acme)" --evaluate

# Regroup the saved clips without decoding audio again, e.g. after deciding
# two labels are one person:
deps/mlx-runtime-venv/bin/python scripts/enroll-voiceprints.py rebuild \
    --same-person "Dana=Dana (Acme),Dee" --evaluate

curl -F file=@meeting.m4a -F diarize=true -F response_format=verbose_json \
     http://127.0.0.1:8014/v1/audio/transcriptions
```

The server re-reads `profiles.json` whenever it changes, so there's no restart
after re-enrolling. `identify=false` turns naming off for a single request.

`speakers[]` in the response gains `name` (the label to display), `voice_match`
(the profile), `voice_score`, and the `runner_up` with its score. So an
unnamed speaker still tells you who it sounded most like and by how much.
Segments gain `speaker_name`. The `text`, `srt` and `vtt` formats write the
name where there is one.

### One voice, two names

`--same-person "Dana=Dana (Acme)"` builds one profile from both
labels, since they're the same voice. The name is then chosen by who else is in
the recording: if another identified speaker is "… (Acme)", it's a work
call and the name is "Dana (Acme)". Otherwise it's "Dana".

## Settings

| key | meaning |
|---|---|
| `MLX_VOICEPRINT_MODEL_PATH` | WeSpeaker weights; blank disables naming |
| `MLX_VOICEPRINT_PROFILES` | `models/voiceprints/profiles.json` |
| `MLX_VOICEPRINT_THRESHOLD` | similarity a voice needs to be named; blank = what `--evaluate` chose |
| `MLX_VOICEPRINT_MARGIN` | how far the best match must beat the runner-up; blank = what `--evaluate` chose |

## Choosing the threshold and margin

`--evaluate` runs leave-one-recording-out. For each recording, every profile is
rebuilt without it, and each person in it is looked up among everyone else,
both with all their clips and with only 3 (someone who barely spoke). People
enrolled from a single recording can't be recognised, so they stand in for
strangers: any name they get is wrong.

Every lookup is then replayed through the matching rule over a grid of
thresholds and margins. The pair kept names the most people correctly while
wrong names (a known person misnamed, or a stranger named at all) stay under
2% of lookups. Being left unnamed is not counted as wrong: the transcript keeps
`speaker_N`, which a person can correct.

The margin is what handles two labels that are one voice. If the same person
is enrolled under two names, every lookup of them is a near-tie between the
two, and the margin leaves them unnamed. `profiles.json` records how similar
every pair of profiles is to each other. Any pair above ~0.8 is worth checking,
and is either one person (merge them with `--same-person`) or a mislabelled
recording.

## Verifying the embedding

The MLX weights come from `mlx-community/wespeaker-voxceleb-resnet34-LM`, but
its bundled `resnet_embedding.py` is **wrong**. It gives the convolutions a
bias the checkpoint doesn't have, and it flattens the pooled statistics in a
different order from PyTorch. Its embeddings score about 0.0 cosine against
Wespeaker's own ONNX export on the same input: noise. The network in
`speaker_embedding.py` matches that ONNX export to 1.0000 on identical
features. With mlx-audio's Kaldi fbank in place of torchaudio's, it scores
0.989–0.997. The remaining gap is float32 noise in near-silent bands, once
the log floor is set to torchaudio's (float32 eps, not 1e-8).

## Measured

On a personal MacWhisper library of about 140 transcripts: 3,529 clips from 40
people after merging the labels that were one voice (47 labels before).

| | all of a speaker's clips | 3 clips |
|---|---|---|
| right person ranked first | 96.5% | 94.2% |
| named, and right | 90.7% | 90.7% |
| named, and wrong | 1.8% | 1.8% |

(threshold 0.47, margin 0.04; 86 recognition lookups and 23 strangers.)

Individual clips of the same person across recordings score a median of 0.71,
and clips of different people score 0.05. Profile-to-profile, unrelated people
top out around 0.53. Every pair above 0.8 turned out to be one voice under two
labels, or one recording with its labels mixed up.

End to end on recordings **left out of enrollment**:

- **A 2-hour, 4-person meeting:** all four were named correctly, and each
  winner beat the runner-up by 0.4–0.6. The work label was chosen from the
  others present. The whole request (transcription, diarization, naming)
  took 2 min 55 s.
- **A 45-minute, 5-person call:** three were named correctly. The longest
  speaker had never been enrolled and stayed unnamed (0.34). One stayed
  unnamed on a tie (0.772 against 0.770) with a profile that turned out to
  be built from his own mislabelled lines in another recording.
