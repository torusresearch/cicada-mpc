import logging
logging.basicConfig(level=logging.INFO)

import math, numpy

from cicada.communicator import SocketCommunicator
from cicada.logging import Logger
from cicada import encoding
from cicada.active import ActiveArrayShare, ActiveProtocolSuite
from cicada.additive import AdditiveArrayShare
from cicada.shamir import ShamirArrayShare

ORDER = 115792089237316195423570985008687907852837564279074904382605163141518161494337
LOG2Q = 12
LOG2P = 8
DIM = 512
BITS = ((2**LOG2Q - 1)**2 * DIM).bit_length() # Maximum integer size.
KAPPA = 40 # Statistical security level.
L = math.ceil((ORDER.bit_length() + KAPPA) / LOG2P) # Number of p-bit elements required to fill up the target range.

def key_gen(protocol: ActiveProtocolSuite) -> ActiveArrayShare:
    (bits, vals) = protocol.random_bitwise_secret(bits=LOG2Q, shape=(DIM,))
    return vals

def evaluate(
    protocol: ActiveProtocolSuite,
    key: ActiveArrayShare,
    hx: numpy.ndarray,
) -> ActiveArrayShare:
    communicator = protocol.communicator
    log = Logger(logger=logging.getLogger(), communicator=communicator)
    log.info(f"Player {communicator.rank}: before compute r = a * k")
    
    # r = a * k
    def field_dot(x, y):
        """Vector dot product with support for plaintext values."""
        result = protocol.field_multiply(x, y)
        return protocol.sum(result)
    r = list(field_dot(hxi, key) for hxi in hx)
    r = ActiveArrayShare((
        AdditiveArrayShare(numpy.stack([ri.additive.storage for ri in r])),
        ShamirArrayShare(numpy.stack([ri.shamir.storage for ri in r])),
    ))
    log.info(f"Player {communicator.rank}: after compute r = a * k = {r}")
    
    # r = round(r)
    r = round(protocol, r)
    
    return compose(protocol, r)
    
def round(
    protocol: ActiveProtocolSuite,
    x: ActiveArrayShare,
) -> ActiveArrayShare:
    return trunc(protocol, x, BITS, LOG2Q-LOG2P, LOG2Q)
    # x_prime = trunc_top(protocol, x, BITS, LOG2Q)
    # return trunc_bot(protocol, x_prime, BITS, LOG2Q-LOG2P)
    
    # bits = protocol.bit_decompose(x, bits=BITS)    
    # slice = bits[:, LOG2Q-LOG2P:LOG2Q]
    # return protocol.bit_compose(slice)

def trunc(
    protocol: ActiveProtocolSuite,
    a: ActiveArrayShare,
    k: int,
    m1: int,
    m2: int,
) -> ActiveArrayShare:
    """
    Truncate the top and bottom bits of the `k`-bit integer `a`. Returns the
    integer corresponding to the bits in the range `[m1:m2]` (least-significant
    bit first).
    """
    communicator = protocol.communicator
    log = Logger(logger=logging.getLogger(), communicator=communicator)
    
    log.info(f"Player {communicator.rank}: before prandm")
    log.info(f"Player {communicator.rank}: a = {a}")
    storage: numpy.ndarray = a.additive.storage
    r_dprime, r_prime, r = prandm(protocol, k, m2, shape=storage.shape)
    log.info(f"Player {communicator.rank}: after prandm")
    
    # c = (2^(k-1) + a + 2^m2 * r_dprime + r_prime).reveal()
    two_pow_ksub1 = protocol.field.full_like(storage, 2**(k-1))
    two_pow_m2 = protocol.field.full_like(storage, 2**m2)
    log.info(f"Player {communicator.rank}: before compute twopow_add_a")
    log.info(f"Player {communicator.rank}: a.shape = {storage.shape}")
    log.info(f"Player {communicator.rank}: two_pow_m2.shape = {two_pow_m2.shape}")
    c = protocol.field_add(two_pow_m2, a)
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
    
    def slice_r(m) -> ActiveArrayShare:
        return ActiveArrayShare((
            AdditiveArrayShare(numpy.array(r.additive.storage[:, -m:], dtype=r.additive.storage.dtype)),
            ShamirArrayShare(numpy.array(r.shamir.storage[:, -m:], dtype=r.shamir.storage.dtype)),
        ))

    # Compute separate c_prime for m1 and m2.
    c1_prime = protocol.field.full_like(a.additive.storage, c % 2**m1)
    c2_prime = protocol.field.full_like(a.additive.storage, c % 2**m2)
    u1 = public_bitwise_less_than(protocol, lhspub=c1_prime, rhs=slice_r(m1))
    u2 = public_bitwise_less_than(protocol, lhspub=c2_prime, rhs=slice_r(m2))
    # u1 = protocol.share(src=0, secret=numpy.full_like(a.additive.storage, 0), shape=a.additive.storage.shape)
    # u2 = protocol.share(src=0, secret=numpy.full_like(a.additive.storage, 0), shape=a.additive.storage.shape)
    
    log.info(f"Player {communicator.rank}: after ltl")
    # log.info(f"Player {communicator.rank}: after ltl: u1 = {protocol.reveal(u1)}, u2 = {protocol.reveal(u2)}")
    log.info(f"Player {communicator.rank}: after ltl: c1_prime = {c1_prime}, c2_prime = {c2_prime}")
    
    # a1_prime = c1_prime - bit_compose(r[:m1]) + u1 * 2^m1
    two_pow_m1 = protocol.field.full_like(a.additive.storage, 2**m1)
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
    two_pow_m1_inv = protocol.field.full_like(a.additive.storage, invert(2**m1))
    d = protocol.field_multiply(
        protocol.field_subtract(a2_prime, a1_prime),
        two_pow_m1_inv,
    )
    log.info(f"Player {communicator.rank}: after compute d: d = {protocol.reveal(d)}")
    return d
    
def trunc_top(
    protocol: ActiveProtocolSuite,
    a: ActiveArrayShare,
    k: int,
    m: int,
) -> ActiveArrayShare:
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
    two_pow_ksub1 = protocol.field.full_like(a.additive.storage, 2**(k-1))
    two_pow_m = protocol.field.full_like(a.additive.storage, 2**m)
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
    c_prime = protocol.field.full_like(a.additive.storage, c % 2**m)
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
    protocol: ActiveProtocolSuite,
    a: ActiveArrayShare,
    k: int,
    m: int,
) -> ActiveArrayShare:
    """
    Truncate the bottom bits of the `k`-bit integer `a`. Returns the integer
    corresponding to the bits in the range `[m:]` (least-significant bit first).
    """
    communicator = protocol.communicator
    log = Logger(logger=logging.getLogger(), communicator=communicator)
    
    a_prime = trunc_top(protocol, a, k, m)
    invert = lambda x: pow(x, protocol.field.order - 2, protocol.field.order)
    two_pow_m_inv = protocol.field.full_like(a.additive.storage, invert(2**m))
    d = protocol.field_multiply(
        protocol.field_subtract(a, a_prime),
        two_pow_m_inv,
    )
    
    a_clear = protocol.reveal(a)[0]
    trunc_a = int(bin(a_clear)[2:][:-m], 2)
    log.info(f"Player {communicator.rank}: a = {a_clear} = {bin(a_clear)}, trunc_bot(a) = {trunc_a} = {bin(trunc_a)}")
    
    return d
    
def prandm(protocol: ActiveProtocolSuite, k: int, m: int, shape: list[int]) -> (ActiveArrayShare, ActiveArrayShare, ActiveArrayShare):
    # r_dprime = k + KAPPA - m
    _, r_dprime = protocol.random_bitwise_secret(bits=k + KAPPA - m, shape=shape)
    r, r_prime = protocol.random_bitwise_secret(bits=m, shape=shape)
    return r_dprime, r_prime, r

def compose(
    protocol: ActiveProtocolSuite,
    a: ActiveArrayShare
) -> ActiveArrayShare:    
    log2_p1 = LOG2Q - LOG2P
    log2_p2 = 2 * LOG2P - LOG2Q
    def shift(i):
        p1_shift = (i - 1) * log2_p1
        p2_shift = i * log2_p2
        return max(0, p1_shift + p2_shift)
    
    scalars = list(1 << shift(i) for i in range(a.additive.storage.shape[0]))
    scalars = protocol.field(numpy.array(scalars))
    
    prod = protocol.field_multiply(a, scalars)
    return protocol.sum(prod)

def public_bitwise_less_than(protocol: ActiveProtocolSuite, *, lhspub: numpy.ndarray, rhs: ActiveArrayShare) -> ActiveArrayShare:    
    if lhspub.shape != rhs.additive.storage.shape[:-1]:
        raise ValueError('rhs is not of the expected shape - it should be the same as lhs except the last dimension') # pragma: no cover
    bitwidth = rhs.additive.storage.shape[-1]
    lhsbits = []
    for val in lhspub:
        tmplist = [int(x) for x in bin(val)[2:]]
        if len(tmplist) < bitwidth:
            tmplist = [0 for x in range(bitwidth-len(tmplist))] + tmplist
        lhsbits.append(tmplist)
    lhsbits = numpy.array(lhsbits, dtype=protocol.field.dtype)
    assert(lhsbits.shape == rhs.additive.storage.shape)
    one = numpy.array(1, dtype=protocol.field.dtype)
    flatlhsbits = lhsbits.reshape((-1, lhsbits.shape[-1]))
    flatrhsbits = ActiveArrayShare((
        AdditiveArrayShare(rhs.additive.storage.reshape((-1, rhs.additive.storage.shape[-1]))),
        ShamirArrayShare(rhs.shamir.storage.reshape((-1, rhs.shamir.storage.shape[-1]))),
    ))
    results=[]
    for j in range(len(flatlhsbits)):
        xord = []
        preord = []
        msbdiff=[]
        rhs_bit_at_msb_diff = []
        for i in range(bitwidth):
            rhsbit = ActiveArrayShare((
                AdditiveArrayShare(storage=numpy.array(flatrhsbits.additive.storage[j,i], dtype=protocol.field.dtype)),
                ShamirArrayShare(storage=numpy.array(flatrhsbits.shamir.storage[j,i], dtype=protocol.field.dtype))
            ))
            if flatlhsbits[j][i] == 1:
                xord.append(protocol.field_subtract(lhs=one, rhs=rhsbit))
            else:
                xord.append(rhsbit)
        preord = [xord[0]]
        for i in range(1, bitwidth):
            preord.append(protocol.logical_or(lhs=preord[i-1], rhs=xord[i]))
        msbdiff = [preord[0]]
        for i in range(1,bitwidth):
            msbdiff.append(protocol.field_subtract(lhs=preord[i], rhs=preord[i-1]))
        for i in range(bitwidth):
            rhsbit = ActiveArrayShare((
                AdditiveArrayShare(storage=numpy.array(flatrhsbits.additive.storage[j,i], dtype=protocol.field.dtype)),
                ShamirArrayShare(storage=numpy.array(flatrhsbits.shamir.storage[j,i], dtype=protocol.field.dtype))
            ))
            rhs_bit_at_msb_diff.append(protocol.field_multiply(rhsbit, msbdiff[i]))
        result = rhs_bit_at_msb_diff[0]
        for i in range(1,bitwidth):
            result = protocol.field_add(lhs=result, rhs=rhs_bit_at_msb_diff[i])
        results.append(result)
    return ActiveArrayShare((
        AdditiveArrayShare(numpy.array([x.additive.storage for x in results], dtype=protocol.field.dtype).reshape(rhs.additive.storage.shape[:-1])),
        ShamirArrayShare(numpy.array([x.shamir.storage for x in results], dtype=protocol.field.dtype).reshape(rhs.shamir.storage.shape[:-1]))
    ))

def main(communicator: SocketCommunicator):
    log = Logger(logger=logging.getLogger(), communicator=communicator)
    protocol = ActiveProtocolSuite(
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