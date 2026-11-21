"""EveryVoice OpenBible fine-tunes (multilingual-tts/EveryVoice-OpenBible-<Language>, FastSpeech2 + HiFi-GAN, 22.05 kHz,
single narrator, CC-BY-SA-4.0; checkpoints are model version 1.2 and need everyvoice 0.4.x). Follows the model-card snippet; all sentences go through one synthesize_helper call,
then the wavs are moved into the common layout. Env: Multilingual-MOS-everyvoice (everyvoice 0.4.1, torch 2.7.1).
    python src/synthesis/everyvoice_tts.py --language Yoruba
"""
import inspect, os, shutil, sys, tempfile, torch
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__)); from common import *

REPO_IDS = {"Arabic": "multilingual-tts/EveryVoice-OpenBible-Arabic-Standard"}        # others: ...-<Language>
args = parse_args(__doc__.split("\n")[0], lambda p: p.add_argument("--repo-id"))
repo_id = args.repo_id or REPO_IDS.get(args.language, f"multilingual-tts/EveryVoice-OpenBible-{args.language.replace(' ', '-')}")

from huggingface_hub import snapshot_download
from everyvoice.config.type_definitions import DatasetTextRepresentation
try:    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import SynthesizeOutputFormats   # <= 0.4.x
except ImportError: from everyvoice.config.type_definitions import SynthesizeOutputFormats                                   # 0.5.0
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.synthesize import get_global_step, synthesize_helper
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import FastSpeech2
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import load_hifigan_from_checkpoint

local = Path(snapshot_download(repo_id)); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt, voc_ckpt = local / "feature_prediction.ckpt", local / "vocoder.ckpt"
model = FastSpeech2.load_from_checkpoint(str(ckpt)).to(device).eval()
vocoder_model, vocoder_config = load_hifigan_from_checkpoint(torch.load(str(voc_ckpt), map_location=device, weights_only=True), device)
speaker, language = next(iter(model.speaker2id)), next(iter(model.lang2id))
print(f"{repo_id}: speakers {list(model.speaker2id)} languages {list(model.lang2id)} -> using {speaker}/{language}")

sents = load_sentences(args.language, args.limit)
tmp = Path(tempfile.mkdtemp(prefix="everyvoice_", dir="/network/scratch/g/guzmand/tmp"))
if not args.dry_run:
    synthesize_helper(model=model, texts=None, style_reference=None, language=None, speaker=None, duration_control=1.0,
                      global_step=get_global_step(ckpt), output_type=[SynthesizeOutputFormats.wav],
                      text_representation=DatasetTextRepresentation.characters, accelerator=("gpu" if device.type == "cuda" else "cpu"),
                      devices="auto", device=device, batch_size=1, num_workers=1, filelist=None,
                      filelist_data=[dict(basename=r.clip_id, characters=r.text, language=language, speaker=speaker, duration_control=1.0, is_last_input_chunk=True)
                                     for r in sents.itertuples()],
                      output_dir=tmp, teacher_forcing_directory=None, vocoder_model=vocoder_model, vocoder_config=vocoder_config,
                      vocoder_global_step=get_global_step(voc_ckpt),
                      **({'simple_filenames': True} if 'simple_filenames' in inspect.signature(synthesize_helper).parameters else {}))
# the wav writer names files truncate_basename(slugify(text))--speaker--language--...; map them back to clip ids
from everyvoice.utils import slugify
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.utils import truncate_basename
produced = {r.clip_id: next((q for q in (tmp / "wav").glob("*.wav") if q.name.startswith(truncate_basename(slugify(r.text)) + "--")), None)
            for r in sents.itertuples()} if not args.dry_run else {}
missing = [k for k, q in produced.items() if q is None]
if missing: sys.exit(f"EveryVoice produced no wav for {missing}; files: {[q.name for q in (tmp / 'wav').glob('*.wav')][:5]}")

def synth(text, voice):
    clip_id = sents.loc[sents.text == text, "clip_id"].iloc[0]
    wav, sr = sf.read(produced[clip_id], dtype="int16"); return wav, sr

run("EveryVoice", repo_id, args.language, synth, [dict(voice_id=f"{speaker}", gender="unknown")],
    dict(duration_control=1.0, vocoder="HiFi-GAN iSTFT (repo vocoder.ckpt)"), limit=args.limit, overwrite=True, dry_run=args.dry_run)
shutil.rmtree(tmp, ignore_errors=True)
