import numpy as np
from pytorch_lightning import seed_everything
import torchaudio
import torch
import os
import torch.nn.functional as F
from src.utilities.audio import TacotronSTFT
import yaml
from src.latent_diffusion.models.musicldm import MusicLDM, DDPM, DDIMSampler
from einops import repeat
from torch import autocast
import librosa


def pad_wav(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    elif waveform_length < segment_length:
        temp_wav = np.zeros((1, segment_length))
        temp_wav[:, :waveform_length] = waveform
    return temp_wav

def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5

def read_wav_file(filename, segment_length):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, segment_length)

    waveform = waveform / np.max(np.abs(waveform))
    waveform = 0.5 * waveform

    return waveform

def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    log_magnitudes_stft = (
        torch.squeeze(log_magnitudes_stft, 0).numpy().astype(np.float32)
    )
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, log_magnitudes_stft, energy

def _pad_spec(fbank, target_length=1024):
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    if fbank.size(-1) % 2 != 0:
        fbank = fbank[..., :-1]

    return fbank


def wav_to_fbank(filename, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None

    # mixup
    waveform = read_wav_file(filename, target_length * 160)  # hop size is 160

    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

    fbank = torch.FloatTensor(fbank.T)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform


def make_batch_for_text_to_audio(text, waveform=None, fbank=None, batchsize=1):
    text = [text] * batchsize
    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to .")

    if(fbank is None):
        fbank = torch.zeros((batchsize, 1024, 64))  # Not used, here to keep the code format
    else:
        fbank = torch.FloatTensor(fbank)
        fbank = fbank.expand(batchsize, 1024, 64)
        assert fbank.size(0) == batchsize

    stft = torch.zeros((batchsize, 1024, 512))  # Not used

    if(waveform is None):
        waveform = torch.zeros((batchsize, 160000))  # Not used
    else:
        waveform = torch.FloatTensor(waveform)
        waveform = waveform.expand(batchsize, -1)
        assert waveform.size(0) == batchsize

    fname = [""] * batchsize  # Not used

    batch = (
        fbank,
        stft,
        None,
        fname,
        waveform,
        text,
    )
    return batch

##################################################################################################


def get_dict_batch(batch):
    # fbank, log_magnitudes_stft, label_indices, fname, waveform, clip_label, text = batch
    fbank, log_magnitudes_stft, label_indices, fname, waveform, text = batch
    ret = {}

    ret["fbank"] = (
        fbank.to(memory_format=torch.contiguous_format).float()
    )
    ret["stft"] = log_magnitudes_stft.to(
        memory_format=torch.contiguous_format
    ).float()
    # ret["clip_label"] = clip_label.to(memory_format=torch.contiguous_format).float()
    ret["waveform"] = waveform.to(memory_format=torch.contiguous_format).float()
    ret["text"] = list(text)
    ret["fname"] = fname

    return ret




##################################################################################################


def duration_to_latent_t_size(duration):
    return int(duration * 25.6)

def set_cond_audio(latent_diffusion):
    latent_diffusion.cond_stage_key = "waveform"
    latent_diffusion.cond_stage_model.embed_mode="audio"
    return latent_diffusion

def set_cond_text(latent_diffusion):
    latent_diffusion.cond_stage_key = "text"
    latent_diffusion.cond_stage_model.embed_mode="text"
    return latent_diffusion



##################### Text to Audio with local model ########################

def text_to_audio(
    latent_diffusion,
    text,
    original_audio_file_path = None,
    seed=42,
    ddim_steps=200,
    duration=10,
    batchsize=1,
    guidance_scale=2.5,
    n_candidate_gen_per_text=3,
    out_dir = None,
    name = "waveform"
):

    # Set save directory:
    if out_dir is not None:
      latent_diffusion.logger_save_dir = out_dir
      latent_diffusion.logger_project = ""
      latent_diffusion.logger_version = ""

    waveform_save_path = latent_diffusion.get_log_dir()
    os.makedirs(waveform_save_path, exist_ok=True)
    print("Waveform save path: ", waveform_save_path)

    seed_everything(int(seed))
    waveform = None
    if(original_audio_file_path is not None):
        waveform = read_wav_file(original_audio_file_path, int(duration * 102.4) * 160)

    batch = make_batch_for_text_to_audio(text, waveform=waveform, batchsize=batchsize)

    batch = get_dict_batch(batch)
    batch["fname"] = [name]

    latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)

    if(waveform is not None):
        print("Generate audio that has similar content as %s" % original_audio_file_path)
        latent_diffusion = set_cond_audio(latent_diffusion)
    else:
        print("Generate audio using text %s" % text)
        latent_diffusion = set_cond_text(latent_diffusion)

    with torch.no_grad():
        waveform = latent_diffusion.generate_sample(
            [batch],
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_gen=n_candidate_gen_per_text,
            duration=duration,
            name = ""
        )
    #returns full path of the file
    return waveform + name + ".wav"

def round_to_multiple(number, multiple):
  x = multiple * round(number / multiple)
  if x == 0: x = multiple
  return x



############################################## Audo to audio with local model ##########################

def audio_to_audio(   #latent_diffusion, text, duration=10, init_path="", guidance_scale=2.5, random_seed=42, n_candidates=3, steps=200, name="waveform"):
                    latent_diffusion,
                    text,
                    original_audio_file_path = None,
                    seed=42,
                    ddim_steps=200,
                    duration=10,
                    batchsize=1,
                    guidance_scale=2.5,
                    n_candidate_gen_per_text=3,
                    out_dir = None,
                    name = "waveform"
                ):

  # Set save directory:
  if out_dir is not None:
    latent_diffusion.logger_save_dir = out_dir
    latent_diffusion.logger_project = ""
    latent_diffusion.logger_version = ""

  use_ddim = ddim_steps is not None
  waveform_save_path = latent_diffusion.get_log_dir()
  os.makedirs(waveform_save_path, exist_ok=True)
  print("Waveform save path: ", waveform_save_path)


  seed_everything(int(seed))
  if(original_audio_file_path is not None):
      waveform = read_wav_file(original_audio_file_path, int(duration * 102.4) * 160)

  batch = make_batch_for_text_to_audio(text, waveform=waveform, batchsize=batchsize)

  batch = get_dict_batch(batch)
  batch["fname"] = [name]

  latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)


  print("Generate audio that has similar content as %s" % original_audio_file_path)
  latent_diffusion = set_cond_audio(latent_diffusion)


  with torch.no_grad():

    batchs = [batch]
    ddim_steps=ddim_steps
    ddim_eta=1.0
    x_T=None
    n_gen=n_candidate_gen_per_text
    unconditional_guidance_scale=guidance_scale
    unconditional_conditioning=None
    # name=""
    use_plms=False

    # Generate n_gen times and select the best
    # Batch: audio, text, fnames
    assert x_T is None
    try:
        batchs = iter(batchs)
    except TypeError:
        raise ValueError("The first input argument should be an iterable object")

    if use_plms:
        assert ddim_steps is not None

    with latent_diffusion.ema_scope("Plotting"):
      for batch in batchs:

        cond = latent_diffusion.cond_stage_model(batch["waveform"])

        # Generate multiple samples
        batch_size = n_gen
        if cond is not None:
            cond = torch.cat([cond] * n_gen, dim=0)
        # text = text * n_gen

        if unconditional_guidance_scale != 1.0:
            unconditional_conditioning = (
                latent_diffusion.cond_stage_model.get_unconditional_condition(batch_size)
            )

        samples, _ = latent_diffusion.sample_log(
            cond=cond,
            batch_size=batch_size,
            x_T=x_T,
            ddim=use_ddim,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            use_plms=use_plms,
        )
        mel = latent_diffusion.decode_first_stage(samples)

        waveform = latent_diffusion.mel_spectrogram_to_waveform(
            mel, savepath=waveform_save_path, bs=None, name=None, save=False
        )
        waveform = np.nan_to_num(waveform)
        waveform = np.clip(waveform, -1, 1)

        ### Similarity comparison ######

        orig_waveform_batch = torch.cat([batch["waveform"]]*batch_size, dim=0)

        orig_embed_mode = latent_diffusion.cond_stage_model.embed_mode
        latent_diffusion.cond_stage_model.embed_mode = "audio"
        new_audio_emb = latent_diffusion.cond_stage_model(torch.FloatTensor(waveform).squeeze(1).cuda())

        orig_audio_emb = latent_diffusion.cond_stage_model(orig_waveform_batch)
        similarity = F.cosine_similarity(new_audio_emb, orig_audio_emb, dim=2)
        similarity.squeeze()
        latent_diffusion.cond_stage_model.embed_mode = orig_embed_mode

        best_index = []

        max_index = torch.argmax(similarity).item()
        best_index.append(max_index)
        print("Similarity between generated audio and text", similarity)
        print("Choose the following indexes:", best_index)


        waveform = waveform[best_index]

        latent_diffusion.save_waveform(waveform, waveform_save_path, name=[name])

  return waveform_save_path+name+".wav"




##################### Style transfer with local model ########################

import contextlib
import wave

def get_duration(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def get_bit_depth(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        bit_depth = f.getsampwidth() * 8
        return bit_depth

def style_transfer(
    latent_diffusion,
    text,
    original_audio_file_path,
    transfer_strength,
    seed=42,
    duration=10,
    batchsize=1,
    guidance_scale=2.5,
    ddim_steps=200,
    config=None,
    out_dir = None,
    name = "waveform"
):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Set save directory:
    if out_dir is not None:
      latent_diffusion.logger_save_dir = out_dir
      latent_diffusion.logger_project = ""
      latent_diffusion.logger_version = ""

    waveform_save_path = latent_diffusion.get_log_dir()
    os.makedirs(waveform_save_path, exist_ok=True)
    print("Waveform save path: ", waveform_save_path)



    assert original_audio_file_path is not None, "You need to provide the original audio file path"
    
    audio_file_duration = librosa.get_duration(filename=original_audio_file_path) #get_duration(original_audio_file_path)
    
    # assert get_bit_depth(original_audio_file_path) == 16, "The bit depth of the original audio file %s must be 16" % original_audio_file_path
    
    if(transfer_strength >= 1.00):
        print("Warning: The transfer must be from 0 to less than 1. 1 and more will result in Error; Automatically set duration to 0.99 seconds")
        transfer_strength = 0.99
    elif(transfer_strength < 0.00):
        print("Warning: The transfer must be from 0 to less than 1. negative number will result in Error; Automatically set duration to 0.00 seconds")
        transfer_strength = 0.00

    if(duration > audio_file_duration):
        print("Warning: Duration you specified %s-seconds must equal or smaller than the audio file duration %ss" % (duration, audio_file_duration))
        duration = round_to_multiple(audio_file_duration, 2.5)
        print("Set new duration as %s-seconds" % duration)
    else:
       duration = duration
       print("Warning: Audio file is longer then duration you specified %s-seconds. We take duration you specified as input" % (duration))

    latent_diffusion = set_cond_text(latent_diffusion)

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        pass
    #     config = default_audioldm_config()

    seed_everything(int(seed))
    # latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
    latent_diffusion.cond_stage_model.embed_mode = "text"

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    mel, _, _ = wav_to_fbank(
        original_audio_file_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT
    )
    mel = mel.unsqueeze(0).unsqueeze(0).to(device)
    mel = repeat(mel, "1 ... -> b ...", b=batchsize)
    init_latent = latent_diffusion.get_first_stage_encoding(
        latent_diffusion.encode_first_stage(mel)
    )  # move to latent space, encode and sample
    if(torch.max(torch.abs(init_latent)) > 1e2):
        init_latent = torch.clip(init_latent, min=-10, max=10)
    sampler = DDIMSampler(latent_diffusion)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=1.0, verbose=False)

    t_enc = int(transfer_strength * ddim_steps)
    prompts = text

    with torch.no_grad():
      with autocast("cuda"):
        with latent_diffusion.ema_scope():
          uc = None
          if guidance_scale != 1.0:
            uc = latent_diffusion.cond_stage_model.get_unconditional_condition(
                batchsize
            )

          c = latent_diffusion.get_learned_conditioning([prompts] * batchsize)
          z_enc = sampler.stochastic_encode(
            init_latent, torch.tensor([t_enc] * batchsize).to(device)
          )

          samples = sampler.decode(
            z_enc,
            c,
            t_enc,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc,
          )
          # x_samples = latent_diffusion.decode_first_stage(samples) # Will result in Nan in output
          # print(torch.sum(torch.isnan(samples)))
          x_samples = latent_diffusion.decode_first_stage(samples)
          # print(x_samples)
          # x_samples = latent_diffusion.decode_first_stage(samples[:,:,:-3,:])
          # print(x_samples)
          # waveform = latent_diffusion.first_stage_model.decode_to_waveform(
          #   x_samples
          # )


          waveform = latent_diffusion.mel_spectrogram_to_waveform(
              x_samples, savepath=waveform_save_path, bs=None, name=name, save=False
          )
          waveform = np.nan_to_num(waveform)
          waveform = np.clip(waveform, -1, 1)
          waveform = torch.FloatTensor(waveform).to(torch.float32)

          latent_diffusion.save_waveform(waveform, waveform_save_path, name=[name])

    return waveform_save_path+name+".wav"