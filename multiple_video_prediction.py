"""
Running

python ./multiple_video_prediction.py \
    --prop_generator_model_path ./sample/best_prop_model.pt \
    --pretrained_cap_model_path ./sample/best_cap_model.pt \
    --device_id 0 \
    --max_prop_per_vid 100 \
    --nms_tiou_thresh 0.4
"""



import argparse
import os
import sys
import subprocess
import json

import numpy as np
import torch

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from datasets.captioning_dataset import ActivityNetCaptionsDataset
# from datasets.load_features import load_features_from_npy
from datasets.load_features import crop_a_segment, pad_segment
from epoch_loops.captioning_epoch_loops import make_masks
from model.captioning_module import BiModalTransformer
from model.proposal_generator import MultimodalProposalGenerator
from utilities.proposal_utils import (get_corner_coords,
                                      remove_very_short_segments,
                                      select_topk_predictions, trim_proposals, non_max_suppresion)
from epoch_loops.captioning_epoch_loops import greedy_decoder

from typing import Dict, List, Union

class Config(object):
    # I need this to keep the name defined to load the config objects from model checkpoints.
    def __init__(self, to_log=True):
        pass

def load_features_from_npy(
        feature_paths: Dict[str, str], start: float, end: float, duration: float, pad_idx: int,
        device: int, get_full_feat=False, pad_feats_up_to: Dict[str, int] = None
    ) -> Dict[str, torch.Tensor]:
    '''Loads the pre-extracted features from numpy files.
    This function is conceptually close to `datasets.load_feature.load_features_from_npy` but cleaned up
    for demonstration purpose.

    Args:
        feature_paths (Dict[str, str]): Paths to the numpy files (keys: 'audio', 'rgb', 'flow).
        start (float, None): Start point (in secs) of a proposal, if used for captioning the proposals.
        end (float, None): Ending point (in secs) of a proposal, if used for captioning the proposals.
        duration (float): Duration of the original video in seconds.
        pad_idx (int): The index of the padding token in the training vocabulary.
        device (int): GPU id.
        get_full_feat (bool, optional): Whether to output full, untrimmed, feature stacks. Defaults to False.
        pad_feats_up_to (Dict[str, int], optional): If get_full_feat, pad to this value. Different for audio
                                                    and video modalities. Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: A dict holding 'audio', 'rgb' and 'flow' features.
    '''

    # load features. Please see README in the root folder for info on video features extraction
    stack_vggish = np.load(feature_paths['audio'])
    stack_rgb = np.load(feature_paths['rgb'])
    stack_flow = np.load(feature_paths['flow'])

    stack_vggish = torch.from_numpy(stack_vggish).float()
    stack_rgb = torch.from_numpy(stack_rgb).float()
    stack_flow = torch.from_numpy(stack_flow).float()

    # for proposal generation we pad the features
    if get_full_feat:
        stack_vggish = pad_segment(stack_vggish, pad_feats_up_to['audio'], pad_idx)
        stack_rgb = pad_segment(stack_rgb, pad_feats_up_to['video'], pad_idx)
        stack_flow = pad_segment(stack_flow, pad_feats_up_to['video'], pad_idx=0)
    # for captioning use trim the segment corresponding to a prop
    else:
        stack_vggish = crop_a_segment(stack_vggish, start, end, duration)
        stack_rgb = crop_a_segment(stack_rgb, start, end, duration)
        stack_flow = crop_a_segment(stack_flow, start, end, duration)

    # add batch dimension, send to device
    stack_vggish = stack_vggish.to(torch.device(device)).unsqueeze(0)
    stack_rgb = stack_rgb.to(torch.device(device)).unsqueeze(0)
    stack_flow = stack_flow.to(torch.device(device)).unsqueeze(0)

    return {'audio': stack_vggish,'rgb': stack_rgb,'flow': stack_flow}


def load_prop_model(
        device: int, prop_generator_model_path: str, pretrained_cap_model_path: str, max_prop_per_vid: int
    ) -> tuple:
    '''Loading pre-trained proposal generator and config object which was used to train the model.

    Args:
        device (int): GPU id.
        prop_generator_model_path (str): Path to the pre-trained proposal generation model.
        pretrained_cap_model_path (str): Path to the pre-trained captioning module (prop generator uses the
                                         encoder weights).
        max_prop_per_vid (int): Maximum number of proposals per video.

    Returns:
        Config, torch.nn.Module: config, proposal generator
    '''
    # load and patch the config for user-defined arguments
    checkpoint = torch.load(prop_generator_model_path, map_location='cpu')
    cfg = checkpoint['config']
    cfg.device = device
    cfg.max_prop_per_vid = max_prop_per_vid
    cfg.pretrained_cap_model_path = pretrained_cap_model_path
    cfg.train_meta_path = './data/train.csv'  # in the saved config it is named differently

    # load anchors
    anchors = {
        'audio': checkpoint['anchors']['audio'],
        'video': checkpoint['anchors']['video']
    }

    # define model and load the weights
    model = MultimodalProposalGenerator(cfg, anchors)
    device = torch.device(cfg.device)
    torch.cuda.set_device(device)
    model.load_state_dict(checkpoint['model_state_dict'])  # if IncompatibleKeys - ignore
    model = model.to(cfg.device)
    model.eval()

    return cfg, model

def load_cap_model(pretrained_cap_model_path: str, device: int) -> tuple:
    '''Loads captioning model along with the Config used to train it and initiates training dataset
       to build the vocabulary including special tokens.

    Args:
        pretrained_cap_model_path (str): path to pre-trained captioning model.
        device (int): GPU id.

    Returns:
        Config, torch.nn.Module, torch.utils.data.dataset.Dataset: config, captioning module, train dataset.
    '''
    # load and patch the config for user-defined arguments
    cap_model_cpt = torch.load(pretrained_cap_model_path, map_location='cpu')
    cfg = cap_model_cpt['config']
    cfg.device = device
    cfg.pretrained_cap_model_path = pretrained_cap_model_path
    cfg.train_meta_path = './data/train.csv'

    # load train dataset just for special token's indices
    train_dataset = ActivityNetCaptionsDataset(cfg, 'train', get_full_feat=False)

    # define model and load the weights
    model = BiModalTransformer(cfg, train_dataset)
    model = torch.nn.DataParallel(model, [device])
    model.load_state_dict(cap_model_cpt['model_state_dict'])  # if IncompatibleKeys - ignore
    model.eval()

    return cfg, model, train_dataset


def generate_proposals(
        prop_model: torch.nn.Module, feature_paths: Dict[str, str], pad_idx: int, cfg: Config, device: int,
        duration_in_secs: float
    ) -> torch.Tensor:
    '''Generates proposals using the pre-trained proposal model.

    Args:
        prop_model (torch.nn.Module): Pre-trained proposal model
        feature_paths (Dict): dict with paths to features ('audio', 'rgb', 'flow')
        pad_idx (int): A special padding token from train dataset.
        cfg (Config): config object used to train the proposal model
        device (int): GPU id
        duration_in_secs (float): duration of the video in seconds. Try this tool to obtain the duration:
            `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 in.mp4`

    Returns:
        torch.Tensor: tensor of size (batch=1, num_props, 3) with predicted proposals.
    '''
    # load features
    feature_stacks = load_features_from_npy(
        feature_paths, None, None, duration_in_secs, pad_idx, device, get_full_feat=True,
        pad_feats_up_to=cfg.pad_feats_up_to
    )

    # form input batch
    batch = {
        'feature_stacks': feature_stacks,
        'duration_in_secs': duration_in_secs
    }

    with torch.no_grad():
        # masking out padding in the input features
        masks = make_masks(batch['feature_stacks'], None, cfg.modality, pad_idx)
        # inference call
        predictions, _, _, _ = prop_model(batch['feature_stacks'], None, masks)
        # (center, length) -> (start, end)
        predictions = get_corner_coords(predictions)
        # sanity-preserving clipping of the start & end points of a segment
        predictions = trim_proposals(predictions, batch['duration_in_secs'])
        # fildering out segments which has 0 or too short length (<0.2) to be a proposal
        predictions = remove_very_short_segments(predictions, shortest_segment_prior=0.2)
        # seƒlect top-[max_prop_per_vid] predictions
        predictions = select_topk_predictions(predictions, k=cfg.max_prop_per_vid)

    return predictions

def caption_proposals(
        cap_model: torch.nn.Module, feature_paths: Dict[str, str],
        train_dataset: torch.utils.data.dataset.Dataset, cfg: Config, device: int, proposals: torch.Tensor,
        duration_in_secs: float
    ) -> List[Dict[str, Union[float, str]]]:
    '''Captions the proposals using the pre-trained model. You must specify the duration of the orignal video.

    Args:
        cap_model (torch.nn.Module): pre-trained caption model. Use load_cap_model() functions to obtain it.
        feature_paths (Dict[str, str]): dict with paths to features ('audio', 'rgb' and 'flow').
        train_dataset (torch.utils.data.dataset.Dataset): train dataset which is used as a vocab and for
                                                          specfial tokens.
        cfg (Config): config object which was used to train caption model. pre-trained model checkpoint has it
        device (int): GPU id to calculate on.
        proposals (torch.Tensor): tensor of size (batch=1, num_props, 3) with predicted proposals.
        duration_in_secs (float): duration of the video in seconds. Try this tool to obtain the duration:
            `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 in.mp4`

    Returns:
        List(Dict(str, Union(float, str))): A list of dicts where the keys are 'start', 'end', and 'sentence'.
    '''

    results = []

    with torch.no_grad():
        for start, end, conf in proposals.squeeze():
            # load features
            feature_stacks = load_features_from_npy(
                feature_paths, start, end, duration_in_secs, train_dataset.pad_idx, device
            )

            # decode a caption for each segment one-by-one caption word
            ints_stack = greedy_decoder(
                cap_model, feature_stacks, cfg.max_len, train_dataset.start_idx, train_dataset.end_idx,
                train_dataset.pad_idx, cfg.modality
            )
            assert len(ints_stack) == 1, 'the func was cleaned to support only batch=1 (validation_1by1_loop)'

            # transform integers into strings
            strings = [train_dataset.train_vocab.itos[i] for i in ints_stack[0].cpu().numpy()]

            # remove starting token
            strings = strings[1:]
            # and remove everything after ending token
            # sometimes it is not in the list (when the caption is intended to be larger than cfg.max_len)
            try:
                first_entry_of_eos = strings.index('</s>')
                strings = strings[:first_entry_of_eos]
            except ValueError:
                pass

            # join everything together
            sentence = ' '.join(strings)
            # Capitalize the sentence
            sentence = sentence.capitalize()

            # add results to the list
            results.append({
                'start': round(start.item(), 1),
                'end': round(end.item(), 1),
                'sentence': sentence
            })

    return results

def which_ffprobe() -> str:
    '''Determines the path to ffprobe library
    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffprobe'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffprobe_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffprobe_path

def get_video_duration(path):
    '''Determines the duration of the custom video
    Returns:
        float -- duration of the video in seconds'''
    cmd = f'{which_ffprobe()} -hide_banner -loglevel panic' \
          f' -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {path}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    video_duration = float(result.stdout.decode('utf-8').replace('\n', ''))
    print('Video Duration:', video_duration)
    return video_duration

import cv2
def get_length(filename):
    video = cv2.VideoCapture(filename)
    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return duration, frame_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiple video prediction')
    parser.add_argument('--prop_generator_model_path', required=True)
    parser.add_argument('--pretrained_cap_model_path', required=True)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--max_prop_per_vid', type=int, default=5)
    parser.add_argument('--nms_tiou_thresh', type=float, help='removed if tiou > nms_tiou_thresh. In (0, 1)')
    args = parser.parse_args()
    
    video_fullpaths = ["/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos274_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos347_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos470_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos427_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos223_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos460_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos282_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos233_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos159_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos407_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos440_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos093_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos213_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos320_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos263_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos161_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos285_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos340_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos447_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos094_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos328_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos457_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos204_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos106_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos380_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos369_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos074_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos419_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos178_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos013_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos429_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos181_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos359_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos301_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos411_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos388_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos252_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos235_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos170_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos225_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos127_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos421_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos134_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos007_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos288_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos465_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos229_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos422_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos298_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos114_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos394_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos335_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos038_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos402_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos104_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos153_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos089_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos362_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos462_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos355_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos269_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos415_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos113_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos218_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos115_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos029_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos071_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos208_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos250_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos088_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos237_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos135_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos162_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos016_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos314_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos172_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos299_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos296_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos343_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos011_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos122_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos291_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos049_x264.mp4", "/home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos434_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion001_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary001_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault001_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson001_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest001_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism001_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting014_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting013_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting012_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting001_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing013_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing014_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing012_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing015_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing011_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents001_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery011_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery001_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery013_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery003_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery012_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting010_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting007_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting008_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting001_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting009_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting006_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting005_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting002_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting004_x264.mp4", "/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting003_x264.mp4"]

    videos_names = [
        'Normal_Videos263_x264', 'Assault009_x264', 'Robbery001_x264', 'Burglary008_x264', 'Arson009_x264', 'Shooting001_x264', 'Normal_Videos427_x264',
        'Normal_Videos411_x264', 'Explosion010_x264', 'Vandalism007_x264', 'Normal_Videos252_x264', 'Normal_Videos340_x264', 'RoadAccidents002_x264',
        'Normal_Videos388_x264', 'Arrest002_x264', 'Normal_Videos049_x264', 'Burglary006_x264', 'Arson007_x264', 'Vandalism009_x264',
        'Vandalism006_x264', 'Abuse003_x264', 'Arson008_x264', 'Burglary009_x264', 'Shooting010_x264', 'Normal_Videos421_x264', 'Normal_Videos402_x264', 
        'Normal_Videos106_x264', 'Vandalism008_x264', 'Normal_Videos127_x264', 'Arson006_x264', 'Explosion001_x264', 'Arrest003_x264', 
        'Burglary007_x264', 'Shoplifting003_x264', 'Normal_Videos285_x264', 'Normal_Videos233_x264', 'Arson005_x264', 'Explosion002_x264', 
        'Normal_Videos422_x264', 'Abuse010_x264', 'Normal_Videos460_x264', 'Normal_Videos250_x264', 'Normal_Videos394_x264', 'Normal_Videos178_x264', 
        'Normal_Videos159_x264', 'Burglary004_x264', 'Normal_Videos298_x264', 'Normal_Videos074_x264', 'Normal_Videos115_x264', 'Shooting003_x264', 
        'Normal_Videos038_x264', 'Normal_Videos007_x264', 'Normal_Videos223_x264', 'Vandalism005_x264', 'Normal_Videos134_x264', 'Normal_Videos288_x264', 
        'Arrest010_x264', 'Normal_Videos093_x264', 'Normal_Videos296_x264', 'Robbery003_x264', 'Normal_Videos181_x264', 'Normal_Videos299_x264', 
        'Burglary005_x264', 'Normal_Videos362_x264', 'Normal_Videos440_x264', 'Arrest001_x264', 'Normal_Videos343_x264', 'Shoplifting001_x264', 
        'Normal_Videos301_x264', 'Normal_Videos104_x264', 'Normal_Videos029_x264', 'Normal_Videos016_x264', 'Normal_Videos213_x264', 
        'Normal_Videos320_x264', 'Robbery002_x264', 'Fighting004_x264', 'Normal_Videos135_x264', 'Shooting002_x264', 'Normal_Videos114_x264', 
        'Abuse001_x264', 'Explosion005_x264', 'Abuse009_x264', 'Arson002_x264', 'RoadAccidents001_x264', 'Burglary003_x264', 'Arrest007_x264', 
        'Normal_Videos161_x264', 'Assault002_x264', 'Shooting004_x264', 'Normal_Videos328_x264', 'Abuse007_x264', 'Vandalism002_x264', 'Normal_Videos094_x264', 
        'Normal_Videos291_x264', 'Normal_Videos355_x264', 'Robbery004_x264', 'Arrest009_x264', 'Fighting002_x264', 'Burglary002_x264', 'Normal_Videos447_x264', 
        'Arrest006_x264', 'Normal_Videos269_x264', 'Normal_Videos011_x264', 'Normal_Videos235_x264', 'Arson003_x264', 'Normal_Videos122_x264', 'Abuse008_x264', 
        'Explosion004_x264', 'Robbery005_x264', 'Fighting003_x264', 'Arrest008_x264', 'Normal_Videos170_x264', 'Normal_Videos457_x264', 
        'Vandalism003_x264', 'Shooting005_x264', 'Normal_Videos113_x264', 'Normal_Videos434_x264', 'Abuse006_x264', 'Normal_Videos204_x264', 'Assault010_x264', 
        'Normal_Videos088_x264', 'Abuse005_x264', 'Explosion009_x264', 'Shooting006_x264', 'Arson010_x264', 'Normal_Videos429_x264', 'Normal_Videos347_x264', 
        'Normal_Videos274_x264', 'Normal_Videos071_x264', 'Burglary001_x264', 'Normal_Videos359_x264', 'Explosion007_x264', 'Vandalism010_x264', 
        'Normal_Videos218_x264', 'Normal_Videos335_x264', 'Vandalism001_x264', 'Explosion008_x264', 'Abuse004_x264', 'Normal_Videos314_x264', 'Shooting007_x264', 
        'Normal_Videos380_x264', 'Normal_Videos369_x264', 'Normal_Videos153_x264', 'Normal_Videos172_x264', 'Normal_Videos419_x264', 
        'Normal_Videos013_x264', 'Normal_Videos229_x264', 'Explosion006_x264', 'Normal_Videos407_x264', 'Arson001_x264', 'Normal_Videos208_x264', 
        'Normal_Videos237_x264', 'Stealing002_x264', 'Arrest004_x264', 'Assault001_x264', 'Normal_Videos162_x264', 'Shoplifting004_x264', 'Normal_Videos282_x264', 
    ]
    feature_paths = []
    for v in videos_names:
        feature_path = {
            'audio': '/home1/ndat/566/video_captioning/BMT/feature_cache/ucf_crime/vggish/' + v + '_vggish.npy',
            'rgb': '/home1/ndat/566/video_captioning/BMT/feature_cache/ucf_crime/i3d/' + v + '_rgb.npy',
            'flow': '/home1/ndat/566/video_captioning/BMT/feature_cache/ucf_crime/i3d/' + v + '_flow.npy',
        }
        feature_paths.append(feature_path)
        
    print('Loading model....')
    # Loading models and other essential stuff
    cap_cfg, cap_model, train_dataset = load_cap_model(args.pretrained_cap_model_path, args.device_id)
    prop_cfg, prop_model = load_prop_model(
        args.device_id, args.prop_generator_model_path, args.pretrained_cap_model_path, args.max_prop_per_vid
    )
    
    print('Captioning....')
    output = {}
    with open('./output_with_timestamp.json', 'w') as output_file:
        N = len(feature_paths)
        for i in range(N):
            duration = -1
            for f in video_fullpaths:
                if videos_names[i] in f:
                    duration, _ = get_length(f)
                    duration = int(duration)
                    break
                
            # Proposal
            proposals = generate_proposals(
                prop_model, feature_paths[i], train_dataset.pad_idx, prop_cfg, args.device_id, duration    
            )
            # NMS if specified
            if args.nms_tiou_thresh is not None:
                proposals = non_max_suppresion(proposals.squeeze(), args.nms_tiou_thresh)
                proposals = proposals.unsqueeze(0)
            # Captions for each proposal
            captions = caption_proposals(
                cap_model, feature_paths[i], train_dataset, cap_cfg, args.device_id, proposals, duration
            )
            print(videos_names[i], ' . '.join([x['sentence'] for x in captions]))
            output[videos_names[i]] = captions
            #output_file.write(videos_names[i] + '||' +  ' . '.join([x['sentence'] for x in captions]) + '\n') 
        output_file.write(json.dumps(output))
        output_file.close()

