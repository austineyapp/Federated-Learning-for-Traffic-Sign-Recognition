from torch import distributed as dist

from grace_dl.dist import Communicator


class Allreduce(Communicator):

    def send_receive(self, tensors, name, ctx):
        for tensor_compressed in tensors:
            dist.all_reduce(tensor_compressed)
            self.size += tensor_compressed.element_size() * tensor_compressed.nelement()
            if self.compressor.average:
                tensor_compressed.div_(self.world_size)
        return self.compressor.decompress(tensors, ctx)

    def acc(self, uncompressed_tensor):
        self.uncompressed_size += uncompressed_tensor.element_size() * uncompressed_tensor.nelement()

    def printr(self):
        print("Uncompressed")
        print(self.uncompressed_size)
        print("Compressed")
        print(self.size)
        print("Data Volume Compression Ratio: {:.1f}x".format(self.uncompressed_size/self.size))