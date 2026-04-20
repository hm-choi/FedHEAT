#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.manifold import TSNE

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List


from servers.build import SERVER_REGISTRY
from servers.he_engine import *

@SERVER_REGISTRY.register()
class Server():

    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        return local_weights
    

@SERVER_REGISTRY.register()
class ServerM(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        self.global_delta = global_delta
        self.global_momentum = global_momentum


    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        if self.args.server.momentum>0:

            if not self.args.server.get('FedACG'): 
                for param_key in local_weights:               
                    local_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]
                    
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
            

        return local_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    
    
    t = 0
    t1 = 130
    t2 = 1370
    log_degree = 6 # for invSqrt initial value
    iteration = 4 # for invSqrt newton's method
    A, B = 10**-5, 7 * (10**-1) # for invSqrt scaling
    m_level = 12
    v_level = 12

    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + 1 * (10**-7)

        self.global_delta = flatten_to_numpy(global_delta)
        self.global_momentum = flatten_to_numpy(global_momentum)
        self.global_v = flatten_to_numpy(global_v)

    def encrypted_aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        # index setting
        self.t += 1
        C = len(client_ids)

        i_for_m = 1 + (self.t - 1) % self.t1
        j_for_v = 1 + (self.t - 1) % self.t2

        beta1 = self.args.server.momentum
        beta2 = self.args.server.beta

        # update learning rate
        server_lr = copy.copy(self.args.trainer.global_lr)
        server_lr *= (beta1 ** i_for_m) / (beta2 ** (j_for_v/2))

        # update tau
        server_tau = copy.copy(self.args.server.tau)
        server_tau /= beta2 ** j_for_v
        
        ####### Debug lr, tau #######
        print("t:", self.t)
        print("i:", i_for_m)
        print("j:", j_for_v)
        print("beta1:", beta1)
        print("beta2:", beta2)
        print("lr(before):", self.args.trainer.global_lr)
        print("tau(before):", self.args.server.tau)
        print("lr(after):", server_lr)
        print("tau(after):", server_tau)

        # heaan setting
        log_slots = 15
        num_slots = 2 ** log_slots
        context, sk, pk, ect, dct, evt, bts, dt = heaan_setting(log_slots)

        # encrypt server's m, v
        global_momentum = self.global_momentum
        global_v = self.global_v

        # encryption
        chunk_size = num_slots
        encrypted_global_momentum = []
        encrypted_global_v = []

        for i in range(0, len(global_momentum), chunk_size):
            chunk_momentum = global_momentum[i:i + chunk_size]
            chunk_v = global_v[i:i + chunk_size]

            if len(chunk_momentum) < chunk_size:
                chunk_momentum = pad(chunk_momentum, chunk_size, np.float64)
                chunk_v = pad(chunk_v, chunk_size, np.float64)

            temp_momentum = hn.Ciphertext(context)
            msg_momentum = hn.Message(chunk_momentum)
            msg_momentum.to(dt)
            ect.encrypt(msg_momentum, pk, temp_momentum, self.m_level)
            encrypted_global_momentum.append(temp_momentum)
            msg_momentum.to_host()

            temp_v = hn.Ciphertext(context)
            msg_v = hn.Message(chunk_v)
            msg_v.to(dt)
            ect.encrypt(msg_v, pk, temp_v, self.v_level)
            encrypted_global_v.append(temp_v)
            msg_v.to_host()

        # add constant to v
        if self.t <= 600 or self.t % 5 == 0:
            k = 1 * (10**-8)
            add(evt, encrypted_global_v, k, encrypted_global_v)

        # ####### Debug v #######
        # decrypted_temp = []
        # for i in range(len(encrypted_global_v)):
        #     temp = hn.Message(log_slots)
        #     dct.decrypt(encrypted_global_v[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print(f"v(before add delta and tau):{decrypted_temp.min()} ({np.argmin(decrypted_temp)}) {decrypted_temp.max()} ({np.argmax(decrypted_temp)})")

        # encrypt each client's delta respectively
        param_order = list(model_dict.keys())
        num_clients = len(local_deltas[param_order[0]])

        client_chunks = [[] for _ in range(num_clients)]
        param_meta = [] # for restroing the model structure

        for key in param_order:
            ref_tensor = model_dict[key]
            numel = ref_tensor.numel()
            param_meta.append((key, ref_tensor.shape, numel))

            for client_idx in range(num_clients):
                delta_tensor = local_deltas[key][client_idx]
                client_chunks[client_idx].append(delta_tensor.reshape(-1).cpu())

        client_delta_vectors = [
            torch.cat(chunks, dim=0).detach().cpu().numpy() for chunks in client_chunks
        ]

        # encrypt and compute sum of delta_t
        sum = []
        for i, vec in enumerate(client_delta_vectors):
            chunk_size = num_slots
            for j in range(0, len(vec), chunk_size):
                chunk = vec[j:j + chunk_size]
                
                # 0-padding
                if len(chunk) < chunk_size:
                    padded = np.zeros(chunk_size, dtype=vec.dtype)
                    padded[:len(chunk)] = chunk
                    chunk = padded

                m = hn.Message(chunk)
                ct = hn.Ciphertext(context)
                m.to(dt)
                ect.encrypt(m, pk, ct)
                m.to_host()

                if i == 0:
                    sum.append(ct)
                else:
                    evt.add(sum[j//chunk_size], ct, sum[j//chunk_size])
                    ct.to_host()
            print(f"encryption for client {i} success: num_ct", len(sum))
        
        # ####### Debug sum #######
        # decrypted_temp = []
        # for i in range(len(sum)):
        #     temp = hn.Message(log_slots)
        #     dct.decrypt(sum[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("sum:", decrypted_temp.min(), decrypted_temp.max())

        # compute delta_t for m using pre-normalization
        delta_m = [hn.Ciphertext(context) for _ in range(len(sum))]
        mult(evt, sum, (1-beta1)/(beta1**i_for_m)/C, delta_m)

        # ####### Debug delta_m #######
        # decrypted_temp = []
        # for i in range(len(delta_m)):
        #     temp = hn.Message(log_slots)
        #     dct.decrypt(delta_m[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("delta_m:", decrypted_temp.min(), decrypted_temp.max())
        # print("m_denom", (1-beta1)/(beta1**i_for_m)/C)

        # compute delta_t for v using pre-normalization
        delta_v = [hn.Ciphertext(context) for _ in range(len(sum))]
        v_denom = (1-beta2)/(beta2**j_for_v) # for applying equation of v
        v_denom /= C**2 # for mean
        mult(evt, sum, v_denom**0.5, delta_v)
        square(evt, delta_v, delta_v)

        # ####### Debug delta_v #######
        # decrypted_temp = []
        # for i in range(len(delta_v)):
        #     temp = hn.Message(log_slots)
        #     dct.decrypt(delta_v[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("delta_v:", decrypted_temp.min(), decrypted_temp.max())
        # print("v_denom", v_denom)

        # update m, v
        if i_for_m == 1 and self.t > 1:
            mult(evt, encrypted_global_momentum, beta1**self.t1, encrypted_global_momentum)
            self.m_level -= 1
        if j_for_v == 1 and self.t > 1:
            mult(evt, encrypted_global_v, beta2**self.t2, encrypted_global_v)
            self.v_levellevel -= 1
        
        add(evt, encrypted_global_momentum, delta_m, encrypted_global_momentum) # update m
        add(evt, encrypted_global_v, delta_v, encrypted_global_v) # update v

        # decrypt updated delta, m, v and upload it to server
        decrypted_global_delta = []
        decrypted_global_momentum = []
        decrypted_global_v = []

        for i in range(len(sum)):
            temp_delta = hn.Message(log_slots)
            dct.decrypt(sum[i], sk, temp_delta)
            temp_delta.to_host()
            chunk = np.array(temp_delta, dtype=np.float64) / C
            decrypted_global_delta = np.concatenate([decrypted_global_delta, chunk])

            temp_momentum = hn.Message(log_slots)
            dct.decrypt(encrypted_global_momentum[i], sk, temp_momentum)
            temp_momentum.to_host()
            chunk = np.array(temp_momentum, dtype=np.float64)
            decrypted_global_momentum = np.concatenate([decrypted_global_momentum, chunk])

            temp_v = hn.Message(log_slots)
            dct.decrypt(encrypted_global_v[i], sk, temp_v)
            temp_v.to_host()
            chunk = np.array(temp_v, dtype=np.float64)
            decrypted_global_v = np.concatenate([decrypted_global_v, chunk])

        self.global_delta = decrypted_global_delta[:len(sum)]
        self.global_momentum = decrypted_global_momentum[:len(global_momentum)]
        self.global_v = decrypted_global_v[:len(global_v)]
        print("m:", self.global_momentum.min(), self.global_momentum.max())
        print("v:", self.global_v.min(), self.global_v.max())
        print("delta:", self.global_delta.min(), self.global_delta.max())

        mult(evt, encrypted_global_momentum, server_lr, encrypted_global_momentum) # mult lr
        add(evt, encrypted_global_v, server_tau, encrypted_global_v) # add tau

        # ####### Debug encrypted_global_v #######
        # decrypted_temp = []
        # for i in range(len(encrypted_global_v)):
        #     temp = hn.Message(log_slots)
        #     dct.decrypt(encrypted_global_v[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("v(after add delta and tau):", decrypted_temp.min(), decrypted_temp.max())

        ct = [hn.Ciphertext(context) for _ in range(len(encrypted_global_v))]
        mult(evt, encrypted_global_v, 1/2, ct)
        scaled_ct = [hn.Ciphertext(context) for _ in range(len(encrypted_global_v))]
        mult(evt, encrypted_global_v, 2/(self.B-self.A), scaled_ct)
        sub(evt, scaled_ct, (self.B + self.A)/(self.B-self.A), scaled_ct)
        
        # invSqrt(v+tau)
        ret = HEInvSqrt(context, evt, bts, ct, scaled_ct, 2**self.log_degree-1, self.iteration, self.A, self.B)

        ####### Debug invSqrt #######
        decrypted_temp = []
        for i in range(len(ret)):
            temp = hn.Message(15)
            dct.decrypt(ret[i], sk, temp)
            temp.to_host()
            chunk = np.array(temp, dtype=np.float64)
            decrypted_temp = np.concatenate([decrypted_temp, chunk])
        print(f"invSqrt:{decrypted_temp.min()} ({np.argmin(decrypted_temp)}) {decrypted_temp.max()} ({np.argmax(decrypted_temp)})")

        sub(evt, encrypted_global_v, server_tau, encrypted_global_v)
        decrypted_temptemp = []
        for i in range(len(encrypted_global_v)):
            temp = hn.Message(15)
            dct.decrypt(encrypted_global_v[i], sk, temp)
            temp.to_host()
            chunk = np.array(temp, dtype=np.float64)
            decrypted_temptemp = np.concatenate([decrypted_temptemp, chunk])
        print(f"at the input:{decrypted_temptemp[np.argmin(decrypted_temp)]} {decrypted_temptemp[np.argmax(decrypted_temp)]}")

        # compute opt step
        mult(evt, encrypted_global_momentum, ret, ret)

        # ####### Debug lr*m #######
        # decrypted_temp = []
        # for i in range(len(encrypted_global_momentum)):
        #     temp = hn.Message(15)
        #     dct.decrypt(encrypted_global_momentum[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("lr*m:", decrypted_temp.min(), decrypted_temp.max())

        # ####### Debug opt step #######
        # decrypted_temp = []
        # for i in range(len(ret)):
        #     temp = hn.Message(15)
        #     dct.decrypt(ret[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("opt step:", decrypted_temp.min(), decrypted_temp.max())
        
        # decrypt optimization step
        opt_step = np.empty(len(ret)*num_slots, dtype=np.float64)
        offset = 0
        for i in range(len(ret)):
            temp = hn.Message(log_slots)
            dct.decrypt(ret[i], sk, temp)
            ret[i].to_host()
            temp.to_host()
            temp = np.array(temp, dtype=np.float64)

            opt_step[offset:offset+num_slots] = temp
            offset += num_slots
        
        # return updated model
        restored = {}
        start = 0

        for key, shape, numel in param_meta:
            chunk = opt_step[start:start + numel]
            tensor = torch.from_numpy(chunk).reshape(shape)
            tensor = tensor.to(device=model_dict[key].device, dtype=model_dict[key].dtype)
            restored[key] = tensor
            start += numel

        for param_key in model_dict.keys():
            model_dict[param_key] += restored[param_key]

        for i in range(len(delta_m)):
            sum[i].to_host()
            encrypted_global_momentum[i].to_host()
            encrypted_global_v[i].to_host()
            delta_m[i].to_host()
            delta_v[i].to_host()
            ct[i].to_host()
            scaled_ct[i].to_host()

        delta_stats = self.compute_stats(self.global_delta)
        m_stats = self.compute_stats(self.global_momentum)
        v_stats = self.compute_stats(self.global_v)

        stats = {
            "delta": delta_stats,
            "m": m_stats,
            "v": v_stats,
        }

        return model_dict, stats

    def compute_stats(self, vec):

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


class ServerAda(Server):    
    
    t = 0
    t1 = 130
    t2 = 1370
    log_degree = 6 # for invSqrt initial value
    iteration = 4 # for invSqrt newton's method
    A, B = 10**-5, 7 * (10**-1) # for invSqrt scaling
    m_level = 12
    v_level = 12

    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + 1 * (10**-7)

        self.global_delta = flatten_to_numpy(global_delta)
        self.global_momentum = flatten_to_numpy(global_momentum)
        self.global_v = flatten_to_numpy(global_v)

    def encrypted_aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        # index setting
        self.t += 1
        C = len(client_ids)

        i_for_m = 1 + (self.t - 1) % self.t1

        beta1 = self.args.server.momentum
        beta2 = self.args.server.beta

        # update learning rate
        server_lr = copy.copy(self.args.trainer.global_lr)
        server_lr *= (beta1 ** i_for_m)

        # update tau
        server_tau = copy.copy(self.args.server.tau)
        
        ####### Debug lr, tau #######
        print("t:", self.t)
        print("i:", i_for_m)
        print("beta1:", beta1)
        print("beta2:", beta2)
        print("lr(before):", self.args.trainer.global_lr)
        print("tau(before):", self.args.server.tau)
        print("lr(after):", server_lr)
        print("tau(after):", server_tau)

        # heaan setting
        log_slots = 15
        num_slots = 2 ** log_slots
        context, sk, pk, ect, dct, evt, bts, dt = heaan_setting(log_slots)

        # encrypt server's m, v
        global_momentum = self.global_momentum
        global_v = self.global_v

        # encryption
        chunk_size = num_slots
        encrypted_global_momentum = []
        encrypted_global_v = []

        for i in range(0, len(global_momentum), chunk_size):
            chunk_momentum = global_momentum[i:i + chunk_size]
            chunk_v = global_v[i:i + chunk_size]

            if len(chunk_momentum) < chunk_size:
                chunk_momentum = pad(chunk_momentum, chunk_size, np.float64)
                chunk_v = pad(chunk_v, chunk_size, np.float64)

            temp_momentum = hn.Ciphertext(context)
            msg_momentum = hn.Message(chunk_momentum)
            msg_momentum.to(dt)
            ect.encrypt(msg_momentum, pk, temp_momentum, self.m_level)
            encrypted_global_momentum.append(temp_momentum)
            msg_momentum.to_host()

            temp_v = hn.Ciphertext(context)
            msg_v = hn.Message(chunk_v)
            msg_v.to(dt)
            ect.encrypt(msg_v, pk, temp_v, self.v_level)
            encrypted_global_v.append(temp_v)
            msg_v.to_host()

        # add constant to v
        if self.t <= 600 or self.t % 3 == 0:
            k = 1 * (10**-8)
            add(evt, encrypted_global_v, k, encrypted_global_v)

        # ####### Debug v #######
        # decrypted_temp = []
        # for i in range(len(encrypted_global_v)):
        #     temp = hn.Message(log_slots)
        #     dct.decrypt(encrypted_global_v[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print(f"v(before add delta and tau):{decrypted_temp.min()} ({np.argmin(decrypted_temp)}) {decrypted_temp.max()} ({np.argmax(decrypted_temp)})")

        # encrypt each client's delta respectively
        param_order = list(model_dict.keys())
        num_clients = len(local_deltas[param_order[0]])

        client_chunks = [[] for _ in range(num_clients)]
        param_meta = [] # for restroing the model structure

        for key in param_order:
            ref_tensor = model_dict[key]
            numel = ref_tensor.numel()
            param_meta.append((key, ref_tensor.shape, numel))

            for client_idx in range(num_clients):
                delta_tensor = local_deltas[key][client_idx]
                client_chunks[client_idx].append(delta_tensor.reshape(-1).cpu())

        client_delta_vectors = [
            torch.cat(chunks, dim=0).detach().cpu().numpy() for chunks in client_chunks
        ]

        # encrypt and compute sum of delta_t
        sum = []
        for i, vec in enumerate(client_delta_vectors):
            chunk_size = num_slots
            for j in range(0, len(vec), chunk_size):
                chunk = vec[j:j + chunk_size]
                
                # 0-padding
                if len(chunk) < chunk_size:
                    padded = np.zeros(chunk_size, dtype=vec.dtype)
                    padded[:len(chunk)] = chunk
                    chunk = padded

                m = hn.Message(chunk)
                ct = hn.Ciphertext(context)
                m.to(dt)
                ect.encrypt(m, pk, ct)
                m.to_host()

                if i == 0:
                    sum.append(ct)
                else:
                    evt.add(sum[j//chunk_size], ct, sum[j//chunk_size])
                    ct.to_host()
            print(f"encryption for client {i} success: num_ct", len(sum))
        
        # ####### Debug sum #######
        # decrypted_temp = []
        # for i in range(len(sum)):
        #     temp = hn.Message(log_slots)
        #     dct.decrypt(sum[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("sum:", decrypted_temp.min(), decrypted_temp.max())

        # compute delta_t for m using pre-normalization
        delta_m = [hn.Ciphertext(context) for _ in range(len(sum))]
        mult(evt, sum, (1-beta1)/(beta1**i_for_m)/C, delta_m)

        # ####### Debug delta_m #######
        # decrypted_temp = []
        # for i in range(len(delta_m)):
        #     temp = hn.Message(log_slots)
        #     dct.decrypt(delta_m[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("delta_m:", decrypted_temp.min(), decrypted_temp.max())
        # print("m_denom", (1-beta1)/(beta1**i_for_m)/C)

        # compute delta_t for v using pre-normalization
        delta_v = [hn.Ciphertext(context) for _ in range(len(sum))]
        v_denom /= C**2 # for mean
        mult(evt, sum, v_denom**0.5, delta_v)
        square(evt, delta_v, delta_v)

        # ####### Debug delta_v #######
        # decrypted_temp = []
        # for i in range(len(delta_v)):
        #     temp = hn.Message(log_slots)
        #     dct.decrypt(delta_v[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("delta_v:", decrypted_temp.min(), decrypted_temp.max())
        # print("v_denom", v_denom)

        # update m
        if i_for_m == 1 and self.t > 1:
            mult(evt, encrypted_global_momentum, beta1**self.t1, encrypted_global_momentum)
            self.m_level -= 1
        
        add(evt, encrypted_global_momentum, delta_m, encrypted_global_momentum) # update m
        add(evt, encrypted_global_v, delta_v, encrypted_global_v) # update v

        # decrypt updated delta, m, v and upload it to server
        decrypted_global_delta = []
        decrypted_global_momentum = []
        decrypted_global_v = []

        for i in range(len(sum)):
            temp_delta = hn.Message(log_slots)
            dct.decrypt(sum[i], sk, temp_delta)
            temp_delta.to_host()
            chunk = np.array(temp_delta, dtype=np.float64) / C
            decrypted_global_delta = np.concatenate([decrypted_global_delta, chunk])

            temp_momentum = hn.Message(log_slots)
            dct.decrypt(encrypted_global_momentum[i], sk, temp_momentum)
            temp_momentum.to_host()
            chunk = np.array(temp_momentum, dtype=np.float64)
            decrypted_global_momentum = np.concatenate([decrypted_global_momentum, chunk])

            temp_v = hn.Message(log_slots)
            dct.decrypt(encrypted_global_v[i], sk, temp_v)
            temp_v.to_host()
            chunk = np.array(temp_v, dtype=np.float64)
            decrypted_global_v = np.concatenate([decrypted_global_v, chunk])

        self.global_delta = decrypted_global_delta[:len(sum)]
        self.global_momentum = decrypted_global_momentum[:len(global_momentum)]
        self.global_v = decrypted_global_v[:len(global_v)]

        mult(evt, encrypted_global_momentum, server_lr, encrypted_global_momentum) # mult lr
        add(evt, encrypted_global_v, server_tau, encrypted_global_v) # add tau

        # ####### Debug encrypted_global_v #######
        # decrypted_temp = []
        # for i in range(len(encrypted_global_v)):
        #     temp = hn.Message(log_slots)
        #     dct.decrypt(encrypted_global_v[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("v(after add delta and tau):", decrypted_temp.min(), decrypted_temp.max())

        ct = [hn.Ciphertext(context) for _ in range(len(encrypted_global_v))]
        mult(evt, encrypted_global_v, 1/2, ct)
        scaled_ct = [hn.Ciphertext(context) for _ in range(len(encrypted_global_v))]
        mult(evt, encrypted_global_v, 2/(self.B-self.A), scaled_ct)
        sub(evt, scaled_ct, (self.B + self.A)/(self.B-self.A), scaled_ct)
        
        # invSqrt(v+tau)
        ret = HEInvSqrt(context, evt, bts, ct, scaled_ct, 2**self.log_degree-1, self.iteration, self.A, self.B)

        ####### Debug invSqrt #######
        decrypted_temp = []
        for i in range(len(ret)):
            temp = hn.Message(15)
            dct.decrypt(ret[i], sk, temp)
            temp.to_host()
            chunk = np.array(temp, dtype=np.float64)
            decrypted_temp = np.concatenate([decrypted_temp, chunk])
        print(f"invSqrt:{decrypted_temp.min()} ({np.argmin(decrypted_temp)}) {decrypted_temp.max()} ({np.argmax(decrypted_temp)})")

        sub(evt, encrypted_global_v, server_tau, encrypted_global_v)
        decrypted_temptemp = []
        for i in range(len(encrypted_global_v)):
            temp = hn.Message(15)
            dct.decrypt(encrypted_global_v[i], sk, temp)
            temp.to_host()
            chunk = np.array(temp, dtype=np.float64)
            decrypted_temptemp = np.concatenate([decrypted_temptemp, chunk])
        print(f"at the input:{decrypted_temptemp[np.argmin(decrypted_temp)]} {decrypted_temptemp[np.argmax(decrypted_temp)]}")

        # compute opt step
        mult(evt, encrypted_global_momentum, ret, ret)

        # ####### Debug lr*m #######
        # decrypted_temp = []
        # for i in range(len(encrypted_global_momentum)):
        #     temp = hn.Message(15)
        #     dct.decrypt(encrypted_global_momentum[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("lr*m:", decrypted_temp.min(), decrypted_temp.max())

        # ####### Debug opt step #######
        # decrypted_temp = []
        # for i in range(len(ret)):
        #     temp = hn.Message(15)
        #     dct.decrypt(ret[i], sk, temp)
        #     temp.to_host()
        #     chunk = np.array(temp, dtype=np.float64)
        #     decrypted_temp = np.concatenate([decrypted_temp, chunk])
        # print("opt step:", decrypted_temp.min(), decrypted_temp.max())
        
        # decrypt optimization step
        opt_step = np.empty(len(ret)*num_slots, dtype=np.float64)
        offset = 0
        for i in range(len(ret)):
            temp = hn.Message(log_slots)
            dct.decrypt(ret[i], sk, temp)
            ret[i].to_host()
            temp.to_host()
            temp = np.array(temp, dtype=np.float64)

            opt_step[offset:offset+num_slots] = temp
            offset += num_slots
        
        # return updated model
        restored = {}
        start = 0

        for key, shape, numel in param_meta:
            chunk = opt_step[start:start + numel]
            tensor = torch.from_numpy(chunk).reshape(shape)
            tensor = tensor.to(device=model_dict[key].device, dtype=model_dict[key].dtype)
            restored[key] = tensor
            start += numel

        for param_key in model_dict.keys():
            model_dict[param_key] += restored[param_key]

        for i in range(len(delta_m)):
            sum[i].to_host()
            encrypted_global_momentum[i].to_host()
            encrypted_global_v[i].to_host()
            delta_m[i].to_host()
            delta_v[i].to_host()
            ct[i].to_host()
            scaled_ct[i].to_host()

        delta_stats = self.compute_stats(self.global_delta)
        m_stats = self.compute_stats(self.global_momentum)
        v_stats = self.compute_stats(self.global_v)

        stats = {
            "delta": delta_stats,
            "m": m_stats,
            "v": v_stats,
        }

        return model_dict, stats

    def compute_stats(self, vec):

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


@SERVER_REGISTRY.register()
class ServerDyn(Server):    
    
    def set_momentum(self, model):
        #global_momentum is h^t in FedDyn paper
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])


        self.global_delta = global_delta
        self.global_momentum = global_momentum

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in self.global_momentum:
            self.global_momentum[param_key] -= self.args.client.Dyn.alpha / self.args.trainer.num_clients * sum(local_deltas[param_key])
            local_weights[param_key] = sum(local_weights[param_key])/C - 1/self.args.client.Dyn.alpha * self.global_momentum[param_key]
        return local_weights