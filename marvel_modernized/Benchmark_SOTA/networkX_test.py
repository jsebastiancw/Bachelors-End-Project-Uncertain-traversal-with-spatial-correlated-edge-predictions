import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from algorithms.CTD import Trainer, CTD
from algorithms.utils.cutom_env import *
from algorithms import func
from algorithms.PQL import QL
from algorithms.GP3 import *
from copy import deepcopy
import time
from algorithms.ActorCritic import ActorCritic

map1 = MapInfo("D:/SE-GAC-master/nn/Anaheim/Anaheim_network.csv")
print(map1.G.edges)