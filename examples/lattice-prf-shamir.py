import logging
logging.basicConfig(level=logging.INFO)

import math, numpy

from cicada.communicator import SocketCommunicator
from cicada.logging import Logger
from cicada import encoding
import cicada.shamir
from cicada.shamir import ShamirArrayShare, ShamirProtocolSuite

ORDER = 115792089237316195423570985008687907852837564279074904382605163141518161494337
LOG2Q = 12
LOG2P = 8
DIM = 512
BITS = ((2**LOG2Q - 1)**2 * DIM).bit_length() # Maximum integer size.
KAPPA = 40 # Statistical security level.
L = math.ceil((ORDER.bit_length() + KAPPA) / LOG2P) # Number of p-bit elements required to fill up the target range.

def key_gen(protocol: ShamirProtocolSuite) -> ShamirArrayShare:
    (bits, vals) = protocol.random_bitwise_secret(bits=LOG2Q, shape=(DIM,))
    return vals

def evaluate(
    protocol: ShamirProtocolSuite,
    key: ShamirArrayShare,
    hx: numpy.ndarray,
) -> ShamirArrayShare:
    # r = a * k
    def field_dot(x, y):
        """Vector dot product with support for plaintext values."""
        result = protocol.field_multiply(x, y)
        return protocol.sum(result)
    r = list(field_dot(hxi, key).storage for hxi in hx)
    r = ShamirArrayShare(numpy.stack(r))
    
    # r = round(r)
    r = round(protocol, r)
    
    return compose(protocol, r)
    
def round(
    protocol: ShamirProtocolSuite,
    x: ShamirArrayShare,
) -> ShamirArrayShare:
    return trunc(protocol, x, BITS, LOG2Q-LOG2P, LOG2Q)
    # x_prime = trunc_top(protocol, x, BITS, LOG2Q)
    # return trunc_bot(protocol, x_prime, BITS, LOG2Q-LOG2P)
    
    # bits = protocol.bit_decompose(x, bits=BITS)    
    # slice = bits[:, LOG2Q-LOG2P:LOG2Q]
    # return protocol.bit_compose(slice)

def trunc(
    protocol: ShamirProtocolSuite,
    a: ShamirArrayShare,
    k: int,
    m1: int,
    m2: int,
) -> ShamirArrayShare:
    """
    Truncate the top and bottom bits of the `k`-bit integer `a`. Returns the
    integer corresponding to the bits in the range `[m1:m2]` (least-significant
    bit first).
    """
    communicator = protocol.communicator
    log = Logger(logger=logging.getLogger(), communicator=communicator)
    
    r_dprime, r_prime, r = prandm(protocol, k, m2, shape=a.storage.shape)
    
    # c = (2^(k-1) + a + 2^m2 * r_dprime + r_prime).reveal()
    two_pow_ksub1 = protocol.field.full_like(a.storage, 2**(k-1))
    two_pow_m2 = protocol.field.full_like(a.storage, 2**m2)
    log.info(f"Player {communicator.rank}: before compute c")
    c = protocol.field_add(
        two_pow_ksub1,
        protocol.field_add(
            a,
            protocol.field_add(
                protocol.field_multiply(two_pow_m2, r_dprime),
                r_prime,
            ),
        ),
    )
    c = protocol.reveal(c)
    log.info(f"Player {communicator.rank}: after compute c")
    
    def slice_r(m) -> ShamirArrayShare:
        return ShamirArrayShare(numpy.array(r.storage[:, -m:], dtype=r.storage.dtype))

    # Compute separate c_prime for m1 and m2.
    c1_prime = protocol.field.full_like(a.storage, c % 2**m1)
    c2_prime = protocol.field.full_like(a.storage, c % 2**m2)
    u1 = protocol._public_bitwise_less_than(lhspub=c1_prime, rhs=slice_r(m1))
    u2 = protocol._public_bitwise_less_than(lhspub=c2_prime, rhs=slice_r(m2))
    
    log.info(f"Player {communicator.rank}: after ltl")
    log.info(f"Player {communicator.rank}: after ltl: u1 = {protocol.reveal(u1)}, u2 = {protocol.reveal(u2)}")
    log.info(f"Player {communicator.rank}: after ltl: c1_prime = {c1_prime}, c2_prime = {c2_prime}")
    
    # a1_prime = c1_prime - bit_compose(r[:m1]) + u1 * 2^m1
    two_pow_m1 = protocol.field.full_like(a.storage, 2**m1)
    a1_prime = protocol.field_add(
        protocol.field_subtract(
            c1_prime,
            protocol.bit_compose(slice_r(m1))
        ),
        protocol.field_multiply(u1, two_pow_m1),
    )
    # a2_prime = c2_prime - r_prime + u2 * 2^m2
    a2_prime = protocol.field_add(
        protocol.field_subtract(
            c2_prime,
            protocol.bit_compose(slice_r(m2))
        ),
        protocol.field_multiply(u2, two_pow_m2),
    )
    
    a1_prime_clear = protocol.reveal(a1_prime)[0]
    a2_prime_clear = protocol.reveal(a2_prime)[0]
    log.info(f"Player {communicator.rank}: after compute a_prime")
    log.info(f"Player {communicator.rank}: a1_prime = {a1_prime_clear} = {bin(a1_prime_clear)}, a2_prime = {a2_prime_clear} = {bin(a2_prime_clear)}")
    log.info(f"Player {communicator.rank}: a2_prime - a1_prime = {a2_prime_clear - a1_prime_clear} = {bin(a2_prime_clear - a1_prime_clear)}")
    
    a_clear = protocol.reveal(a)[0]
    trunc_a = int(bin(a_clear)[2:][-m2:-m1], 2)
    log.info(f"Player {communicator.rank}: a = {a_clear} = {bin(a_clear)}, trunc(a) = {trunc_a} = {bin(trunc_a)}")
    
    # (a2_prime - a1_prime) / 2^m1
    invert = lambda x: pow(x, protocol.field.order - 2, protocol.field.order)
    two_pow_m1_inv = protocol.field.full_like(a.storage, invert(2**m1))
    d = protocol.field_multiply(
        protocol.field_subtract(a2_prime, a1_prime),
        two_pow_m1_inv,
    )
    log.info(f"Player {communicator.rank}: after compute d: d = {protocol.reveal(d)}")
    return d
    
def trunc_top(
    protocol: ShamirProtocolSuite,
    a: ShamirArrayShare,
    k: int,
    m: int,
) -> ShamirArrayShare:
    """
    Truncate the top bits of the `k`-bit integer `a`. Returns the integer
    corresponding to the bits in the range `[:m]` (least-significant bit first).
    """
    communicator = protocol.communicator
    log = Logger(logger=logging.getLogger(), communicator=communicator)
    
    a_clear = protocol.reveal(a)[0]
    trunc_a = int(bin(a_clear)[2:][-m:], 2)
    log.info(f"Player {communicator.rank}: a = {a_clear} = {bin(a_clear)}, trunc(a) = {trunc_a} = {bin(trunc_a)}")
    
    r_dprime, r_prime, r = prandm(protocol, k, m, shape=a.storage.shape)
    
    # c = (2^(k-1) + a + 2^m2 * r_dprime + r_prime).reveal()
    two_pow_ksub1 = protocol.field.full_like(a.storage, 2**(k-1))
    two_pow_m = protocol.field.full_like(a.storage, 2**m)
    log.info(f"Player {communicator.rank}: before compute c")
    c = protocol.field_add(
        two_pow_ksub1,
        protocol.field_add(
            a,
            protocol.field_add(
                protocol.field_multiply(two_pow_m, r_dprime),
                r_prime,
            ),
        ),
    )
    c = protocol.reveal(c)
    log.info(f"Player {communicator.rank}: after compute c")

    # Compute separate c_prime for m1 and m2.
    c_prime = protocol.field.full_like(a.storage, c % 2**m)
    u = protocol._public_bitwise_less_than(lhspub=c_prime, rhs=r)
    
    log.info(f"Player {communicator.rank}: after ltl")
    log.info(f"Player {communicator.rank}: after ltl: u = {protocol.reveal(u)}")
    log.info(f"Player {communicator.rank}: after ltl: c_prime = {c_prime}")
    
    # a_prime = c_prime - r_prime + u * 2^m
    a_prime = protocol.field_add(
        protocol.field_subtract(
            c_prime,
            r_prime
        ),
        protocol.field_multiply(u, two_pow_m),
    )
    
    a_prime_clear = protocol.reveal(a_prime)[0]
    log.info(f"Player {communicator.rank}: after compute a_prime")
    log.info(f"Player {communicator.rank}: a_prime = {a_prime_clear} = {bin(a_prime_clear)}")
    return a_prime

def trunc_bot(
    protocol: ShamirProtocolSuite,
    a: ShamirArrayShare,
    k: int,
    m: int,
) -> ShamirArrayShare:
    """
    Truncate the bottom bits of the `k`-bit integer `a`. Returns the integer
    corresponding to the bits in the range `[m:]` (least-significant bit first).
    """
    communicator = protocol.communicator
    log = Logger(logger=logging.getLogger(), communicator=communicator)
    
    a_prime = trunc_top(protocol, a, k, m)
    invert = lambda x: pow(x, protocol.field.order - 2, protocol.field.order)
    two_pow_m_inv = protocol.field.full_like(a.storage, invert(2**m))
    d = protocol.field_multiply(
        protocol.field_subtract(a, a_prime),
        two_pow_m_inv,
    )
    
    a_clear = protocol.reveal(a)[0]
    trunc_a = int(bin(a_clear)[2:][:-m], 2)
    log.info(f"Player {communicator.rank}: a = {a_clear} = {bin(a_clear)}, trunc_bot(a) = {trunc_a} = {bin(trunc_a)}")
    
    return d
    
def prandm(protocol: ShamirProtocolSuite, k: int, m: int, shape: list[int]) -> (ShamirArrayShare, ShamirArrayShare, ShamirArrayShare):
    # r_dprime = k + KAPPA - m
    _, r_dprime = protocol.random_bitwise_secret(bits=k + KAPPA - m, shape=shape)
    r, r_prime = protocol.random_bitwise_secret(bits=m, shape=shape)
    return r_dprime, r_prime, r

def compose(
    protocol: ShamirProtocolSuite,
    a: ShamirArrayShare
) -> ShamirArrayShare:    
    log2_p1 = LOG2Q - LOG2P
    log2_p2 = 2 * LOG2P - LOG2Q
    def shift(i):
        p1_shift = (i - 1) * log2_p1
        p2_shift = i * log2_p2
        return max(0, p1_shift + p2_shift)
    
    scalars = list(1 << shift(i) for i in range(a.storage.shape[0]))
    scalars = protocol.field(numpy.array(scalars))
    
    prod = protocol.field_multiply(a, scalars)
    return protocol.sum(prod)

def main(communicator: SocketCommunicator):
    log = Logger(logger=logging.getLogger(), communicator=communicator)
    protocol = cicada.shamir.ShamirProtocolSuite(
        communicator=communicator,
        threshold=2,
        seed=1234,
        order=ORDER,
        encoding=encoding.Identity(),
    )
    
    key = key_gen(protocol)    
    hx = protocol.field(numpy.arange(L * DIM).reshape(L, DIM))
    y = evaluate(protocol, key, hx)
    
    y_revealed = protocol.reveal(y)
    log.info(f"Player {communicator.rank}: y = {y_revealed}")
    import json
    log.info(f"Player {communicator.rank}: communicator.stats = \n{json.dumps(communicator.stats)}")

SocketCommunicator.run(world_size=3, fn=main)