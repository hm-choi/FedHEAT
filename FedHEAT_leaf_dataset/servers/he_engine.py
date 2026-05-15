import heaan as hn
import numpy as np
import os, json

from functools import lru_cache
from typing import Dict, List, Tuple
import re, torch


context = None
sk = None
pk = None
ect = None
dct = None
evt = None
bts = None
dt = None
log_slots = 0


def compute_stats(vec):

    mean = vec.mean()
    std = vec.std()
    min_val = vec.min()
    max_val = vec.max()
    median = np.median(vec)

    skew = ((vec - mean) ** 3).mean() / (std**3 + 1e-12)

    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "median": median,
        "skew": skew,
    }

def flatten_to_numpy(state_dict):
    return torch.cat([v.view(-1) for v in state_dict.values()]).cpu().numpy()

def pad(x, chunk_size):
    if len(x) < chunk_size:
        padded = np.full((chunk_size), 0.00)
        padded[:len(x)] = x
        return padded
    return x

def real_to_complex(vec: np.ndarray, N: int):
    assert vec.ndim == 1
    assert N > 0

    L = vec.shape[0]

    padded_len = ((L + 2 * N - 1) // (2 * N)) * (2 * N)

    if padded_len > L:
        vec = np.pad(vec, (0, padded_len - L), mode='constant')

    blocks = vec.reshape(-1, N)

    complex_blocks = []
    for b in range(0, blocks.shape[0], 2):
        real_part = blocks[b]
        imag_part = blocks[b + 1]
        complex_blocks.append(real_part + 1j * imag_part)

    return np.concatenate(complex_blocks)

def complex_to_real(cvec: np.ndarray, N: int):
    assert cvec.ndim == 1
    assert len(cvec) % N == 0

    blocks = cvec.reshape(-1, N)

    real_blocks = []

    for block in blocks:
        real_blocks.append(block.real)
        real_blocks.append(block.imag)

    return np.concatenate(real_blocks)

def encrypted_real_to_complex(evt, ct, ret):

    num_complex = (len(ct) + 1) // 2

    for i in range(num_complex):
        real_idx = 2 * i
        imag_idx = 2 * i + 1

        if imag_idx >= len(ct):
            z = hn.Ciphertext(ct[real_idx])
            ret[i] = z
            break

        # i * imag
        i_imag = hn.Ciphertext(ct[imag_idx])
        evt.i_mult(i_imag, i_imag)

        # real + i * imag = z
        z = hn.Ciphertext(ct[real_idx])
        evt.add(z, i_imag, z)

        ret[i] = z

        i_imag.to_host()

def encrypted_complex_to_real(evt, ct, ret, length):

    for i in range(len(ct)):

        # real part
        real = hn.Ciphertext(context)
        evt.conjugate(ct[i], real)
        evt.add(ct[i], real, real)

        ret[i*2] = real

        # check dummy imag
        if i*2+1 >= length:
            break
        
        # imaginary part
        imag = hn.Ciphertext(ct[i])
        for _ in range(3):
            evt.i_mult(imag, imag)
        tmp = hn.Ciphertext(imag)
        evt.conjugate(tmp, tmp)
        evt.add(imag, tmp, imag)

        ret[i*2+1] = imag

def debug(dct, sk, ct, log_slots, v_len, name, constant=1.0):
    tmp = np.empty(v_len, dtype=np.float64)
    dec(dct, sk, ct, log_slots, tmp, v_len)
    tmp *= constant
    print(f"{name}(level {ct[0].level}) :{tmp.min()} ({np.argmin(tmp)}) {tmp.max()} ({np.argmax(tmp)})")
    
    return tmp

def heaan_setting():

    global context, sk, pk, ect, dct, evt, bts, dt, log_slots
    if context is None:

        params = hn.ParameterPreset.FGb
        params_preset = str(params)[-3:] + "/"
        heaan_setting_dir_path = "/root/heaan_setting/" + params_preset  ## heaan setting dir path

        device_id = 0 ## GPU id

        context = hn.make_context(params, {device_id})
        dt = hn.Device(hn.DeviceType.GPU, device_id)

        log_slots = 15
        num_slots = 2 ** log_slots

        key_dir_path = heaan_setting_dir_path + "keys/"
        SK_name = "SK"
        PK_name = "PK"

        sk, pk = None, None

        if not (os.path.exists(key_dir_path)):
            os.makedirs(key_dir_path)

        try:
            # key load
            sk = hn.SecretKey(context, key_dir_path + SK_name)
            pk = hn.KeyPack(context, key_dir_path + PK_name)
            
        except:
            # key gen
            sk = hn.SecretKey(context)
            sk.save(key_dir_path + SK_name)

            keygen = hn.KeyGenerator(context, sk)
            keygen.gen_common_keys()
            keygen.gen_rot_keys_for_bootstrap(log_slots)
            keygen.save(key_dir_path + PK_name)
            pk = keygen.keypack

        try:
            sk.to(dt), pk.to(dt)
            ect = hn.Encryptor(context)
            dct = hn.Decryptor(context)
            evt = hn.HomEvaluator(context, pk)
            bts = hn.Bootstrapper(evt)

            m = hn.Message(np.zeros(num_slots))
            m.to(dt)
            c = hn.Ciphertext(context)
            ect.encrypt(m, pk, c)

        except:
            sk.to(dt), pk.to(dt)
            ect = hn.Encryptor(context)
            dct = hn.Decryptor(context)
            evt = hn.HomEvaluator(context, pk)
            bts = hn.Bootstrapper(evt)

        bts.bootstrap(c, c)

    return context, sk, pk, ect, dct, evt, bts, dt, log_slots

def to_host(ct):
    for i in range(len(ct)):
        ct[i].to_host()

def to_dt(ct, dt):
    for i in range(len(ct)):
        ct[i].to(dt)
        
def to_level(evt, ct, level):
    for i in range(len(ct)):
        evt.level_down(ct[i], level, ct[i])

def msg(dt, v, log_slots, m):
    num_slots = 2 ** log_slots
    chunk_size = num_slots

    for i in range(0, len(v), chunk_size):
        chunk = v[i:i+chunk_size]

        if len(chunk) < chunk_size:
            chunk = pad(chunk, chunk_size)

        msg = hn.Message(chunk)
        msg.to(dt)
        m[i//chunk_size] = msg

def enc(ect, pk, dt, v, log_slots, ct, level=12):
    num_slots = 2 ** log_slots
    chunk_size = num_slots

    for i in range(0, len(v), chunk_size):
        chunk = v[i:i+chunk_size]

        if len(chunk) < chunk_size:
            chunk = pad(chunk, chunk_size)

        msg = hn.Message(chunk)
        msg.to(dt)
        ect.encrypt(msg, pk, ct[i//chunk_size], level)

def dec(dct, sk, ct, log_slots, v, length, complex=False):
    pos = 0

    for i in range(len(ct)):
        tmp = hn.Message(log_slots)
        dct.decrypt(ct[i], sk, tmp)
        tmp.to_host()
        if complex is True:
            tmp = np.array(tmp, dtype=np.complex128)
        else:
            tmp = np.array(tmp, dtype=np.float64)

        remain = length - pos
        if remain <= 0:
            break
            
        n = min(tmp.shape[0], remain)
        v[pos:pos+n] = tmp[:n]
        pos += n

def add(evt, ct1, ct2, ret):

    if isinstance(ct2, list):
        for i in range(len(ct1)):
            evt.add(ct1[i], ct2[i], ret[i])
    else:
        for i in range(len(ct1)):
            evt.add(ct1[i], ct2, ret[i])

def sub(evt, ct1, ct2, ret):

    if isinstance(ct2, list):
        for i in range(len(ct1)):
            evt.sub(ct1[i], ct2[i], ret[i])
    else:
        for i in range(len(ct1)):
            evt.sub(ct1[i], ct2, ret[i])

def mult(evt, ct1, ct2, ret):

    if isinstance(ct2, list):
        for i in range(len(ct1)):
            evt.mult(ct1[i], ct2[i], ret[i])
    else:
        for i in range(len(ct1)):
            evt.mult(ct1[i], ct2, ret[i])
            
def integer_mult(evt, ct, k:int, ret):

    for i in range(len(ct)):
        evt.integer_mult(ct[i], k, ret[i])

def square(evt, ct, ret):
    
    for i in range(len(ct)):
        evt.square(ct[i], ret[i])

def bootstrap(bts, ct, is_complex=False):

    for i in range(len(ct)):
        bts.bootstrap(ct[i], ct[i], is_complex)

def next_power_of_two_sqrt(x: int) -> int:
    if x <= 0:
        raise ValueError("x must be a positive integer")

    # k = ceil(log2(x) / 2)
    bitlen = x.bit_length() - 1  # floor(log2(x))
    k = (bitlen + 1) // 2        # ceil(bitlen / 2)

    if (1 << (2 * k)) < x:
        k += 1

    return 1 << k

def cheb_block_transform_a_to_b(a, X):
    """
    a: list/array, a[n] is coefficient of T_n, n=0..N
    X: block size
    returns b as 2D list: b[k][r], where block k, inner index r in [0, X-1]
    ex) b_i = b[k][r] (i = kX+r)
    {b_0 * T_0 + b_1 * T_1      + ... + b_(X-1) * T_(X-1)}  * T_0 +
    {b_X * T_0 + b_(X+1) * T_1  + ... + b_(2X-1) * T_(X-1)} * T_X +
    ...
    = a_0 * T_0 + a_1 * T_1 + ... + a_N * T_N
    """
    N = len(a) - 1
    K = N // X
    c = a[:]  # residual

    # init b
    b = [[0 for _ in range(X)] for _ in range(K + 1)]

    for k in range(K, 0, -1):
        base = k * X

        # r = X-1 down to 1
        for r in range(X - 1, 0, -1):
            h = base + r
            if h <= N:
                b[k][r] = 2 * c[h]
                c[base - r] -= c[h]
                c[h] = 0

        # r = 0
        if base <= N:
            b[k][0] = c[base]
            c[base] = 0

    # k = 0 block
    for r in range(min(X, N + 1)):
        b[0][r] = c[r]
        c[r] = 0

    return b

# A channel monomial is a commutative product of primitive indices.
# Represented as a sorted tuple of ints (ascending) for hashing.
Monomial = Tuple[int, ...]

def mul_monom(a: Monomial, b: Monomial) -> Monomial:
    return tuple(sorted(a + b))

def monom_to_label(m: Monomial) -> str:
    """Display like T16T8T4 (descending indices). Empty monom is T0."""
    if not m:
        return "T0"
    return "".join(f"T{idx}" for idx in sorted(m, reverse=True))

def make_expander(B: int):
    """
    Returns expand(i): dict monomial -> integer coefficient
    representing T_{B*i} as a sum of products of primitives {T_{B*2^k}}
    using: T_{m+n} = 2*T_m*T_n - T_{|m-n|}.
    """
    if not isinstance(B, int) or B <= 0:
        raise ValueError("B must be a positive integer.")

    @lru_cache(maxsize=None)
    def expand(i: int) -> Dict[Monomial, int]:
        if i < 0:
            raise ValueError("i must be non-negative")

        # T_{B*0} = T0 channel
        if i == 0:
            return {(): 1}

        # power of two -> primitive channel T_{B*i}
        if (i & (i - 1)) == 0:
            return {(B * i,): 1}

        # Decompose i = a + b with a = highest power-of-two < i, b = i-a
        a = 1 << (i.bit_length() - 1)
        b = i - a
        if b <= 0:
            return {(B * i,): 1}

        Ea = expand(a)
        Eb = expand(b)
        Ed = expand(a - b)

        out: Dict[Monomial, int] = {}

        # 2 * Ea * Eb
        for ma, ca in Ea.items():
            for mb, cb in Eb.items():
                m = mul_monom(ma, mb)
                out[m] = out.get(m, 0) + 2 * ca * cb

        # - Ed
        for md, cd in Ed.items():
            out[md] = out.get(md, 0) - cd

        # prune zeros
        out = {m: c for m, c in out.items() if c != 0}
        return out

    return expand

def channel_order(B: int, N: int) -> List[Monomial]:
    """
    Output order:
      T0
      T_B
      T_2B, T_2B*T_B
      T_4B, T_4B*T_B, T_4B*T_2B, T_4B*T_2B*T_B
      ...
    """
    if N <= 1:
        return [()]

    max_i = N - 1
    L = max_i.bit_length()
    prim = [B * (1 << k) for k in range(L)]  # B,2B,4B,...

    order: List[Monomial] = []

    # 1) T0 first
    order.append(())

    # 2) remaining channels
    for k, pk in enumerate(prim):
        if k == 0:
            order.append((pk,))
            continue

        for mask in range(0, 1 << k):
            subset = [pk]
            for j in range(k):
                if (mask >> j) & 1:
                    subset.append(prim[j])
            order.append(tuple(sorted(subset)))

    return order

def rewrite_blocks_to_channels(
    blocks: List[List[int]],
    B: int,
    include_zeros: bool = True,
) -> Dict[str, List[int]]:
    """
    Input:
      blocks: N x B list
        blocks[i][j] is the coefficient for T_j inside Block_i.
        (j is in ascending order: T0, T1, ..., T_{B-1})
      B: baby size (must match len(blocks[i]))

    Computes:
      S = sum_{i=0..N-1} Block_i * T_{B*i}
    Rewrites T_{B*i} into channels made of primitives {T_{B*2^k}} and their products,
    and returns:
      dict: channel_label -> coefficient_list_of_length_B
    where each coefficient_list is the (integer-weighted) linear combination of Block rows.
    """
    if not blocks:
        raise ValueError("blocks must be non-empty")
    N = len(blocks)
    if any(len(row) != B for row in blocks):
        raise ValueError("Each block row must have length B")

    expand = make_expander(B)

    collected: Dict[Monomial, List[int]] = {}

    for i in range(N):
        Ei = expand(i)  # monom -> int coeff
        for monom, w in Ei.items():
            if monom not in collected:
                collected[monom] = [0] * B
            for j in range(B):
                collected[monom][j] += w * blocks[i][j]

    out: Dict[str, List[int]] = {}
    for monom in channel_order(B, N):   # T0-first order
        vec = collected.get(monom, [0] * B)
        if include_zeros or any(v != 0 for v in vec):
            out[monom_to_label(monom)] = vec

    return out

def channel_key(name: str) -> Tuple[int, ...]:
    """
    Convert channel name like:
      'T0'          -> (0,)
      'T4'          -> (4,)
      'T8T4'        -> (8,4)
      'T16T8T4'     -> (16,8,4)

    Sorting by this tuple gives ascending-degree order automatically.
    """
    nums = list(map(int, re.findall(r'\d+', name)))
    # T0 should come first
    if nums == [0]:
        return (0,)
    # sort descending inside channel, but ascending across channels
    return tuple(sorted(nums, reverse=True))

def flatten_channels_auto(
    channels: Dict[str, List[int]],
    B: int,
    Degree: int,
) -> List[int]:
    """
    Flatten channels into a 1D list without providing explicit order.
    Order is determined automatically from channel names (ascending degree).

    Missing channels are treated as zero vectors.
    """
    # sort channel names by degree
    names = sorted(channels.keys(), key=channel_key)

    flat: List[int] = []
    for name in names:
        vec = channels[name]
        if len(vec) != B:
            raise ValueError(f"Channel {name} has length {len(vec)}, expected {B}")
        flat.extend(vec)

    return flat[:Degree+1]

def transform_coeffs4BSGS(cbsp, B, deg):
    '''
    ※ B should be a power of two
    Transform coefficients a_i of chebyshev polynomial Σ(a_i*T_i) 
    into a form that is compatible with HEaaN cheb poly eval function
    ''' 
    blocks = cheb_block_transform_a_to_b(cbsp, B)
    channels = rewrite_blocks_to_channels(blocks, B)
    coeffs = flatten_channels_auto(channels, B=B, Degree=deg)

    return coeffs

def ChebysevInvSqrt_ct(evt, bts, scaled_ct, degree:int, A, B, k):
    
    cbsp = np.load(f"./servers/Cheb_coef/Cbsp{degree}_{A:.1e}_{B:.1e}.npy")

    Bstep = next_power_of_two_sqrt(degree)
    cbsp = transform_coeffs4BSGS(cbsp, Bstep, degree)
    hn_cbsp = hn.math.approx.ChebyshevCoefficients(np.array(cbsp), Bstep)

    ret = [hn.math.approx.evaluate_chebyshev_expansion(evt, bts, scaled_ct[i], hn_cbsp, 1/k) for i in range(len(scaled_ct))]

    return ret

def HENewtonInv(context, evt, bts, ct, init_c, iteration:int, k:int, v_len):

    x = ct
    y = init_c
        
    tmp_a = [hn.Ciphertext(context) for _ in range(len(ct))]
    tmp_b = [hn.Ciphertext(context) for _ in range(len(ct))]

    for iter in range(iteration):
                
        mult(evt, y, 3/2, tmp_a)
        mult(evt, x, y, tmp_b)

        square(evt, y, y)
            
        mult(evt, tmp_b, y, tmp_b)
        sub(evt, tmp_a, tmp_b, y)
        
        # debug(dct, sk, y, log_slots, v_len, f"y{iter+1}")

    for i in range(len(y)):
        evt.integer_mult(y[i], k, y[i])
    
    to_host(tmp_a)
    to_host(tmp_b)

    return y

def HEInvSqrt(context, evt, bts, ct, scaled_ct, degree, iteration, A, B, k:int, v_len):
    
    ####### Debug input #######
    # debug(dct, sk, ct, 16, v_len, "input of invSqrt")
    
    ####### Debug scaled_v #######
    # debug(dct, sk, scaled_ct, 16, v_len, "scaled_v")

    init_c = ChebysevInvSqrt_ct(evt, bts, scaled_ct, degree, A, B, k)
    
    ####### Debug cheb eval #######
    debug(dct, sk, init_c, 15, v_len, "init_c")

    ret = HENewtonInv(context, evt, bts, ct, init_c, iteration, k, v_len)

    return ret