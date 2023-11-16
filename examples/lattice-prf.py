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
L = math.ceil((ORDER.bit_length() + KAPPA) / LOG2P) # Number of p-bit elements required to fill up the target range.

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
    r = list(field_dot(hxi, key).storage for hxi in hx)
    r = AdditiveArrayShare(numpy.stack(r))
    
    # r = round(r)
    r = round(protocol, r)
    
    return compose(protocol, r)
    
def round(
    protocol: AdditiveProtocolSuite,
    x: AdditiveArrayShare,
) -> AdditiveArrayShare:
    # return protocol.right_shift(x, bits=LOG2Q)
    return trunc(protocol, x, BITS, LOG2Q-LOG2P, LOG2Q)
    
    # bits = protocol.bit_decompose(x, bits=BITS)    
    # slice = bits[:, LOG2Q-LOG2P:LOG2Q]
    # return protocol.bit_compose(slice)

def trunc(
    protocol: AdditiveProtocolSuite,
    a: AdditiveArrayShare,
    k: int,
    m1: int,
    m2: int,
) -> AdditiveArrayShare:
    """
    Truncate the top and bottom bits of the `k`-bit integer `a`. Returns the
    integer corresponding to the bits in the range `[m1:m2]`.
    """
    (r_dprime, r_prime, r) = prandm(protocol, k, m2)
    # c = (2^(k-1) + a + 2^m2 * r_dprime + r_prime).reveal()
    c = protocol.field_add(
        protocol.field(2**(k-1)),
        protocol.field_add(
            a,
            protocol.field_add(
                protocol.field_multiply(2**m2, r_dprime),
                r_prime,
            ),
        ),
    )
    c = protocol.reveal(c)

    # Compute separate c_prime for m1 and m2.
    c1_prime, c2_prime = c % 2**m1, c % 2**m2
    u1 = protocol._public_bitwise_less_than(c1_prime, r[:m1])
    u2 = protocol._public_bitwise_less_than(c2_prime, r[:m2])
    
    # a1_prime = c1_prime - bit_compose(r[:m1]) + u1 * 2^m1
    a1_prime = protocol.field_add(
        protocol.field_subtract(
            c1_prime,
            protocol.bit_compose(r[:m1])
        ),
        protocol.field_multiply(u1, 2**m1),
    )
    # a2_prime = c2_prime - r_prime + u2 * 2^m2
    a2_prime = protocol.field_add(
        protocol.field_subtract(
            c2_prime,
            protocol.bit_compose(r[:m2])
        ),
        protocol.field_multiply(u2, 2**m2),
    )
    field_inverse = lambda x: pow(x, protocol.field.order - 2, protocol.field.order)
    return protocol.field_multiply(
        protocol.field_subtract(a2_prime, a1_prime),
        field_inverse(2**m1),
    )
    
def prandm(protocol: AdditiveProtocolSuite, k: int, m: int) -> (AdditiveArrayShare, AdditiveArrayShare, AdditiveArrayShare):
    # r_dprime = k + KAPPA - m
    _, r_dprime = protocol.random_bitwise_secret(bits=k + KAPPA - m)
    r, r_prime = protocol.random_bitwise_secret(bits=m)
    return (r_dprime, r_prime, r)

def compose(
    protocol: AdditiveProtocolSuite,
    a: AdditiveArrayShare
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

SocketCommunicator.run(world_size=3, fn=main)