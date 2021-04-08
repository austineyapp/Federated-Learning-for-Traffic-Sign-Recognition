from abc import ABC, abstractmethod


class Memory(ABC):
    @abstractmethod
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        raise NotImplemented("compensate was not implemented.")

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass


class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    @abstractmethod
    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle

class ConvertModel:
    def __init__(self):
        pass

    def model_to_str(self, model):
        data_bytes = pickle.dumps(model)
        data_str = data_bytes.decode('iso-8859-1')
        return data_str

    def str_to_model(self, data_str):
        data_bytes = data_str.encode('iso-8859-1')
        model = pickle.loads(data_bytes)
        return model

import torch
import sys
torch.set_printoptions(threshold=10_000)
class Communicator(ABC):
    @abstractmethod
    def send_receive(self, tensors, name, ctx):
        raise NotImplemented("send was not implemented.")

    def __init__(self, compressor, memory, world_size):
        self.compressor = compressor
        self.memory = memory
        self.world_size = world_size
        self.uncompressed_size = 0
        self.size = 0
        self.converter = ConvertModel()
        self.byteStrSize = 0
        self.uncompressedByteStrSize = 0

    def step(self, tensor, name):
        tensor = self.memory.compensate(tensor, name)
        #tensors_compressed --> tensor, LongTensor
        #ctx --> int, tensor
        tensors_compressed, ctx = self.compressor.compress(tensor, name)
        self.memory.update(tensor, name, self.compressor, tensors_compressed, ctx)
        converted = self.converter.model_to_str(tensors_compressed)
        self.byteStrSize += sys.getsizeof(converted)
        tensors_compressed = self.converter.str_to_model(converted)
        return self.send_receive(tensors_compressed, name, ctx)
