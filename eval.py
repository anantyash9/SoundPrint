import torch
from soundprint.datasets import LibriSpeech
from config import PATH, DATA_PATH
import numpy as np
from soundprint.eval import unseen_speakers_evaluation
import numpy as np
import torch.nn.functional as F
import soundfile as sf
from soundprint.utils import whiten
# unseen_subset = 'dev-clean'
# librispeech_unseen = LibriSpeech(unseen_subset, 3, 4, stochastic=True, pad=False)
model=torch.load('/home/infyblr/voicemap/models/new.pt')
# print (unseen_speakers_evaluation(model, librispeech_unseen, 100))
instance,samperate=sf.read('/home/infyblr/voicemap/reg_emps/738608/738608_3.wav')
instance2,samperate=sf.read('/home/infyblr/voicemap/reg_emps/738608/738608_2.wav')
instance=instance[:48000]
instance = instance[np.newaxis, ::4]
instance = instance[np.newaxis,::]
instance = torch.from_numpy(instance)

instance2=instance2[:48000]
instance2 = instance2[np.newaxis, ::4]
instance2 = instance2[np.newaxis,::]
instance2 = torch.from_numpy(instance2)


x=whiten(instance)
y=whiten(instance2)
x, y = x.cuda(), y.cuda()
with torch.no_grad():
    x_embed = model(x, return_embedding=True)
    y_embed = model(y, return_embedding=True)
    sim = F.cosine_similarity(x_embed, y_embed, dim=1, eps=1e-6)
print(sim.tolist())    
