import logging
logging.basicConfig(level=logging.INFO)

import math, numpy

from cicada.communicator import SocketCommunicator
from cicada.logging import Logger
from cicada import encoding
import cicada.additive
from cicada.additive import AdditiveArrayShare, AdditiveProtocolSuite

ORDER = 115792089237316195423570985008687907852837564279074904382605163141518161494337
LOG2Q = 12
LOG2P = 8
DIM = 512
BITS = ((2**LOG2Q - 1)**2 * DIM).bit_length() # Maximum integer size.
KAPPA = 40 # Statistical security level.
L = 2 # math.ceil((ORDER.bit_length() + KAPPA) / LOG2P) # Number of p-bit elements required to fill up the target range.

def key_gen(protocol: AdditiveProtocolSuite) -> AdditiveArrayShare:
    (bits, vals) = protocol.random_bitwise_secret(bits=LOG2Q, shape=(DIM,))
    return vals

def evaluate(
    protocol: AdditiveProtocolSuite,
    key: AdditiveArrayShare,
    hx: numpy.ndarray,
) -> AdditiveArrayShare:
    # r = a * k
    def field_dot(x, y):
        """Vector dot product with support for plaintext values."""
        result = protocol.field_multiply(x, y)
        return protocol.sum(result)
    r = [field_dot(hxi, key) for hxi in hx]
    # r = AdditiveArrayShare(r)
    
    # r = round(r)
    r = [round(protocol, ri) for ri in r]
    # communicator = protocol.communicator
    # log = Logger(logger=logging.getLogger(), communicator=communicator)
    # for i, ri in enumerate(r):
    #     # log.info(f"r[{i}] (Player {communicator.rank}): {ri}")
    #     ri = protocol.reveal(ri)
    #     log.info(f"r[{i}] revealed (Player {communicator.rank}): {ri}")
    
    return compose(protocol, r)
    
def round(
    protocol: AdditiveProtocolSuite,
    x: AdditiveArrayShare,
) -> AdditiveArrayShare:
    bits = protocol.bit_decompose(x, bits=BITS)
    slice = bits[LOG2Q-LOG2P:LOG2Q]
    return protocol.bit_compose(slice)

def compose(
    protocol: AdditiveProtocolSuite,
    a: list[AdditiveArrayShare]
) -> AdditiveArrayShare:    
    log2_p1 = LOG2Q - LOG2P
    log2_p2 = 2 * LOG2P - LOG2Q
    def shift(i):
        p1_shift = (i - 1) * log2_p1
        p2_shift = i * log2_p2
        return max(0, p1_shift + p2_shift)
    
    acc = protocol.field(0)
    for i, ai in enumerate(a):
        pi = protocol.field(1 << shift(i))
        ai_mul_pi = protocol.field_multiply(ai, pi)
        acc = protocol.field_add(acc, ai_mul_pi)
    return acc

def main(communicator: SocketCommunicator):
    log = Logger(logger=logging.getLogger(), communicator=communicator)
    protocol = cicada.additive.AdditiveProtocolSuite(
        communicator=communicator,
        seed=1234,
        order=ORDER,
        encoding=encoding.Identity(),
    )
    
    key = key_gen(protocol)    
    hx = protocol.field(numpy.arange(L * DIM).reshape(L, DIM))
    y = evaluate(protocol, key, hx)
    
    y_revealed = protocol.reveal(y)
    log.info(f"Output (Player {communicator.rank}): {y_revealed}")

SocketCommunicator.run(world_size=4, fn=main)