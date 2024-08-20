from mmengine.dataset import BaseDataset
from mmdet.registry import DATA_SAMPLERS
from .class_aware_sampler import ClassAwareSampler
import math
from typing import Iterator, Optional, Dict

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS
from .class_aware_sampler import RandomCycleIter

@DATA_SAMPLERS.register_module()
class BalancedShapeSampler(ClassAwareSampler):

    def __init__(self, dataset: BaseDataset, seed: Optional[int] = None, batch_size: int = 2, pairings: Optional[Dict[str, str]] = None,counts={},shuffle=True) -> None:
        assert batch_size % 2 == 0, "Batch size must be even for balanced sampling of two distinct shapes."
        
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = 0

        if seed is None:
            seed = sync_random_seed()
        self.seed = seed

        self.shape_dict = self.get_shape2imgs()
        self.total_size = len(self.dataset)
        self.num_samples = math.ceil(self.total_size / world_size)
        self.pairings = pairings
        self.counts = counts


        # self.pairings = {
        #     "yellow sphere": "brown sphere",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        #     "brown cylinder": "yellow cylinder",
        # }
        # self.pairings = {
        #     "yellow sphere": "brown sphere",
        #     "green cube": "green cylinder",
        #     "red cylinder": "red sphere",
        #     "purple cylinder":"red cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        #     "brown cylinder": "yellow cylinder",
        # }
        # self.pairings = {
        #     "yellow sphere": "brown sphere",
        #     "red sphere":"red cylinder",
        #     "red cylinder":"purple cylinder",
        #     "brown cylinder": "yellow cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        # }
        # self.pairings = {
        #     "yellow sphere": "brown sphere",
        #     "red sphere":"red cylinder",
        #     "green cube": "green cylinder",
        #     "red cylinder":"purple cylinder",
        #     "brown cylinder": "yellow cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        # }
        # self.pairings = {
        #     "yellow sphere": "brown sphere",
        #     "green cube": "green cylinder",
        #     "brown cylinder": "yellow cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        # }
        # self.pairings = {
        #     "brown sphere":"yellow sphere",
        #     "red sphere":"red cylinder",
        #     "red cylinder":"purple cylinder",
        #     "brown cylinder": "yellow cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        # }

        # # self.pairings = {
        # #     "yellow sphere": "brown sphere",
        # #     "red cube": "blue cube",
        # #     "purple sphere": "green sphere",
        # #     "brown cylinder": "yellow cylinder",
        # # }
        # self.counts = counts
        
       # {"red cylinder":0.5}#{"purple cylinder":0.5}#,"yellow cylinder":0.5}
        

        # self.pairings = {
        #     "brown sphere":"yellow sphere",
        #     "brown cylinder": "yellow cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        # }
        # self.count = {}
        
        
        # self.pairings = {
        #     "yellow sphere": "brown sphere",
        #     "brown cylinder": "yellow cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        # }
        # self.pairings = {
        #     "green cube": "green cylinder",
        #     "brown cylinder": "yellow cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        # }
        # self.counts  = {}
        # self.pairings = {
        #     "yellow sphere": "brown sphere",
        #     "red cylinder": "red sphere",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        #     "brown cylinder": "yellow cylinder",
        # }
        #self.counts = {}

        # self.pairings = {
        # #    "yellow sphere": "brown sphere",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        #     "brown cylinder": "yellow cylinder",
        # }
        # self.pairings = {
        #     "red sphere": "red cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        #     "brown cylinder": "yellow cylinder",
        # }
        # self.pairings = {

        #     "red sphere": "red cylinder",
        #     "purple cylinder":"red cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        #     "brown cylinder": "yellow cylinder",
        # }
        # self.pairings = {
        #     "yellow sphere": "brown sphere",
        #     "green cube": "green cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        #     "brown cylinder": "yellow cylinder",
        # }
        # self.pairings = {
        #     "green cube": "green cylinder",
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        #     "brown cylinder": "yellow cylinder",
        # }
        # self.pairings = {
        #     "red cube": "blue cube",
        #     "purple sphere": "green sphere",
        #     "yellow cylinder":"brown cylinder",
        #     "brown sphere": "yellow cylinder",
        #     "brown cylinder":"brown sphere"
        # }
        #self.counts = {"purple cylinder":0.5}#,"yellow cylinder":0.5}
        #self.counts = {}
        # Merge default pairings with user-provided pairings
        #self.pairings = {**default_pairings, **(pairings or {})}

    def get_shape2imgs(self) -> Dict[str, list]:
        """Get a dictionary with shape category as key and image indices as values."""
        shape_dict = {}
        for idx in range(len(self.dataset)):
            img_info = self.dataset[idx]["data_samples"]  # Access image information directly from the dataset
            text = img_info.text
            try:
                label = img_info.gt_instances.labels.item()  
            except Exception:
                continue 
            shape_name = text[label]
            if shape_name not in shape_dict:
                shape_dict[shape_name] = []
            shape_dict[shape_name].append(idx)
        return shape_dict

    def __iter__(self) -> Iterator[int]:
        from math import ceil
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        paired_shapes = list(self.pairings.keys())
        print(paired_shapes)
        keys = list(self.shape_dict.keys())
        indices =[] 
        for key in paired_shapes:
            count = 0
            key2 = self.pairings[key]
            values1 =  self.shape_dict[key] 
            values2 = self.shape_dict[key2]   
            shape1_iter = RandomCycleIter(values1, generator=g)
            shape2_iter = RandomCycleIter(values2, generator=g)
            if key in self.counts.keys():
                break_value = self.counts[key]
                break_value = math.ceil(break_value* len(shape1_iter))

            else:
                break_value = len(shape1_iter)
            while count!=break_value:
#            for i in range(len(shape1_iter)):
                #@indices.append(next(shape2_iter)) 
                indices.append(next(shape1_iter))
                indices.append(next(shape2_iter))
                count+=1



        indices = indices[:]#self.total_size]
        print(len(indices))
        offset = self.num_samples * self.rank
      #  indices = indices[offset:offset + self.num_samples]
        print(len(indices))

        print("Indices:", indices)
        return iter(indices)