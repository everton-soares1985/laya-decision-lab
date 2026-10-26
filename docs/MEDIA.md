# Media and demonstration plan

## Visual identity

Working project name: **Laya Decision Lab**. Tagline: **From multilingual questions
to evidence-grounded decisions.** The visual system uses a dark navy foundation,
teal for evidence/retrieval and blue for model/pipeline structure. Diagrams are
editable SVG and Mermaid, not flattened screenshots of imaginary interfaces.

Current assets:

- `docs/assets/banner.svg`: project identity and system layers.
- `docs/assets/retrieval.svg`: descriptions → embeddings → Qdrant/BM25 → evidence.
- `docs/assets/training.svg`: recorded configuration and correctly scaled 8/12 → 10/12 comparison.
- Architecture, training and mind-map diagrams in Markdown.

All artwork is conceptual and explicitly separate from empirical evidence.

## Original video inventory

Two local recordings are preserved under ignored `Laya_Midia/`:

| Recording | Container | Duration | Approximate file size |
|---|---|---|---|
| 2026-10-20 13-48-07 | MKV | 31 min 1 s | 128.9 MB |
| 2026-10-20 14-42-23 | MKV | 34 min 12 s | 135.1 MB |

Durations and sizes were measured with ffprobe. Their complete contents have
not yet been privacy-reviewed. Timestamps alone do not prove that either video
shows the selected 18:52 training run; recordings must be matched to visible run
IDs before attaching a particular metric or experiment label.

## Reviewed still and experiment matching

Inspection update (26 October): sampled frames and local worker logs distinguish
the two recordings from the later selected experiment:

| Recording / position | Visible evidence | Matching local run |
|---|---|---|
| First, 00:05:00 | Epoch 3 ends at step 51; epoch 4 step 3 at 13:52:34.333 | `full_20261020_134756_311455` (51 train, 204 updates) |
| Second, 00:30:00 | Epoch 4 ends at step 219; completion at 15:07:57; training 1548.69 s | `full_20261020_144147_468978` (219 train, 876 updates) |

The first also has a sampled contact sheet. These checks are not a full review of
every frame. The second recording has an AAC audio track, which has not been
reviewed. No audio or video is approved for public hosting by these checks.

The [reviewed still](../screenshots/training-synthetic-expansion.png) comes from
the second recording at 00:30:00. It shows genuine training output and has not
been reconstructed or edited. The visible 21-item validation result belongs to
that expansion experiment; it is not the final 12-item diagnostic benchmark.
Use it to illustrate the development history, not as proof of the selected run.

Suggested next clip review: inspect the second recording around 00:25:00–00:27:00,
where the recorded training completion occurred. This is a candidate interval,
not an exported or privacy-approved clip. The inspected 00:30:00 still already
shows completed output, so it does not itself establish motion or duration.

## Proposed published sequence

1. **Problem and architecture (15–20 s):** why recognizing a question is not enough.
2. **Actual training excerpt (30–60 s):** visible run configuration, progress and
   checkpoint save, labeled with the correct experiment.
3. **Actual result inspection (20–30 s):** dataset sizes, metrics and limitations.
4. **Grounded-answer demo (30–45 s):** fictitious profile, positive and unsupported
   cases, plus an example with no API calls.

Keep a short overview and a separate longer technical walkthrough. Training
footage does not need to be published as a half-hour unedited desktop recording.

## Privacy and integrity checklist

- Review all selected frames and audio, not only the poster frame.
- Remove personal contacts, tokens, environment files, browser sessions and
  third-party application details before publication.
- Preserve original recordings; derived clips go to a separate output path.
- Label playback speed changes and cuts. Do not imply a long run took seconds.
- Match each clip to its real experiment; do not attach later metrics to an early probe.
- Obtain explicit approval for the final externally hosted uploads.
- Add captions/transcript and a still image with an accurate description.

The full recordings should not be added to normal source commits. A reviewed
external video host or an appropriate release-media mechanism can be selected
at publication time. No public video URLs are claimed yet.
