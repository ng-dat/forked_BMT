#!/bin/bash

cd /home1/ndat/566/video_captioning/BMT/submodules/video_features
python main.py --feature_type i3d --on_extraction save_numpy --device_ids 0 1 --extraction_fps 30 \
--video_paths /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos274_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos347_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos470_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos427_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos223_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos460_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos282_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos233_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos159_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos407_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos440_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos093_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos213_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos320_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos263_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos161_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos285_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos340_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos447_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos094_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos328_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos457_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos204_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos106_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos380_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos369_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos074_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos419_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos178_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos013_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos429_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos181_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos359_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos301_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos411_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos388_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos252_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos235_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos170_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos225_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos127_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos421_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos134_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos007_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos288_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos465_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos229_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos422_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos298_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos114_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos394_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos335_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos038_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos402_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos104_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos153_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos089_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos362_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos462_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos355_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos269_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos415_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos113_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos218_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos115_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos029_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos071_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos208_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos250_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos088_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos237_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos135_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos162_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos016_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos314_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos172_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos299_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos296_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos343_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos011_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos122_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos291_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos049_x264.mp4 /home1/ndat/566/ucf_crime/Training-Normal-Videos-Part-1/Normal_Videos434_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion001_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Explosion/Explosion006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary001_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Burglary/Burglary004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-2/Fighting/Fighting004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault001_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Assault/Assault010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson001_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arson/Arson009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest001_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Arrest/Arrest006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism001_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Vandalism/Vandalism003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting014_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting013_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting012_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting001_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Shoplifting/Shoplifting008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing013_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing014_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing012_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing015_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing011_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-4/Stealing/Stealing010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents001_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery011_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery001_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery013_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery003_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery012_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Robbery/Robbery005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting010_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting007_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting008_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting001_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting009_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting006_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting005_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting002_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting004_x264.mp4 /home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting003_x264.mp4 \
--output_path ../../feature_cache/ucf_crime/i3d
