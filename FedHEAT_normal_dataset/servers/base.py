import copy, math

from utils import *


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
    
    def __init__(self, args):
        super().__init__(args)

        self.t = 0

        self.t1 = args.server.t1
        self.t2 = args.server.t2

        # input domain for invSqrt
        self.A = args.server.A
        self.B = args.server.B

        # deg, #iter for invSqrt
        self.log_degree = args.server.log_degree
        self.iteration = args.server.iteration
        
        self.k = 1
            
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + 1 * (10**-8)

        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v

    def aggregate(self, local_deltas, client_ids, model_dict):
        
        C = len(client_ids)
        
        server_lr = copy.copy(self.args.trainer.global_lr)
        server_tau = copy.copy(self.args.server.tau)
        
        if self.args.server.algo == "original":
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
                self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1-self.args.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])
  
        if self.args.server.algo == "proposed":
            self.t += 1

            i = 1 + (self.t - 1) % self.t1
            j = 1 + (self.t - 1) % self.t2

            server_lr *= (self.args.server.momentum ** i) / (self.args.server.beta ** (j/2))

            server_tau /= self.args.server.beta ** j
            
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                
                if i == 1 and self.t > 1:
                    self.global_momentum[param_key] *= self.args.server.momentum ** self.t1
                if j == 1 and self.t > 1:
                    self.global_v[param_key] *= self.args.server.beta ** self.t2

                self.global_momentum[param_key] += (1-self.args.server.momentum) / (self.args.server.momentum ** i) * self.global_delta[param_key]
                self.global_v[param_key] += (1-self.args.server.beta) / (self.args.server.beta ** j) * (self.global_delta[param_key] * self.global_delta[param_key])
            
        for param_key in model_dict.keys():
            if self.args.server.tau_in == True:
                model_dict[param_key] += server_lr * self.global_momentum[param_key] / ( (self.global_v[param_key] + server_tau) ** 0.5 )
            if self.args.server.tau_in == False:
                model_dict[param_key] += server_lr * self.global_momentum[param_key] / ( (self.global_v[param_key]**0.5)  + server_tau )
            
        delta_stats = compute_stats(flatten_to_numpy(self.global_delta))
        m_stats = compute_stats(flatten_to_numpy(self.global_momentum))
        v_stats = compute_stats(flatten_to_numpy(self.global_v))
        v_add_tau_stats = compute_stats(
            flatten_to_numpy({
                k: v + server_tau for k, v in self.global_v.items()
            })
        )
        
        stats = {
        "delta": delta_stats,
        "m": m_stats,
        "v": v_stats,
        "v_add_tau": v_add_tau_stats,
        }

        return model_dict, stats


    def encrypted_set_momentum(self, model):

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + 1 * (10**-8)

        # heaan setting
        context, sk, pk, ect, dct, evt, bts, dt, log_slots = heaan_setting()
        num_slots = 2 ** log_slots

        # encrypt server's m, v
        global_momentum = flatten_to_numpy(global_momentum)
        global_v = flatten_to_numpy(global_v)

        # encryption
        encrypted_global_momentum = [hn.Ciphertext(context) for _ in range(math.ceil(len(global_momentum)/num_slots))]
        encrypted_global_v = [hn.Ciphertext(context) for _ in range(math.ceil(len(global_v)/num_slots))]

        enc(ect, pk, dt, global_momentum, log_slots, encrypted_global_momentum)
        enc(ect, pk, dt, global_v, log_slots, encrypted_global_v)

        self.global_momentum = encrypted_global_momentum
        self.global_v = encrypted_global_v

    def encrypted_aggregate(self, encrypted_local_deltas, client_ids, weight_len, ):
                
        # index setting
        self.t += 1
        C = len(client_ids)

        i_for_m, j_for_v = 0, 0
        if self.args.server.algo == "proposed":
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
        context, sk, pk, ect, dct, evt, bts, dt, log_slots = heaan_setting()

        # compute sum of delta_t in complex form
        sum_complex = []
        for i in range(len(encrypted_local_deltas)):
            if i == 0:
                for j in range(len(encrypted_local_deltas[i])):
                    sum_complex.append(hn.Ciphertext(encrypted_local_deltas[i][j]))
                to_dt(sum_complex, dt)
            else:
                to_dt(encrypted_local_deltas[i], dt)
                add(evt, sum_complex, encrypted_local_deltas[i], sum_complex)
                to_host(encrypted_local_deltas[i])
            print(f"sum for client {i} success: num_ct", len(sum_complex))

        # bootstrap sum
        if sum_complex[0].level <= 3:
            print(f"Bootstrapping sum(level {sum_complex[0].level})...")
            bootstrap(bts, sum_complex, is_complex=True)

        # convert sum to real value form
        sum_real = [hn.Ciphertext(context) for _ in range(len(self.global_momentum))]
        encrypted_complex_to_real(evt, sum_complex, sum_real, len(self.global_momentum))
        to_host(sum_complex)
        print(f"convert success: num_ct", len(sum_real))
        
        # compute delta_t for m using pre-normalization
        delta_m = [hn.Ciphertext(context) for _ in range(len(sum_real))]
        
        if self.args.server.algo == "original":
            mult(evt, sum_real, (1-beta1)/C/2, delta_m)
            if self.global_momentum[0].level == 3:
                print("Bootstrapping m...")
                bootstrap(bts, self.global_momentum)
            mult(evt, self.global_momentum, beta1, self.global_momentum)
            
        if self.args.server.algo == "proposed":
            mult(evt, sum_real, (1-beta1)/(beta1**i_for_m)/C/2, delta_m)
            if i_for_m == 1 and self.t > 1:
                if (self.global_momentum[0].level == 3):
                    print("Bootstrapping m...")
                    bootstrap(bts, self.global_momentum)
                mult(evt, self.global_momentum, beta1**self.t1, self.global_momentum)

        # update m
        add(evt, self.global_momentum, delta_m, self.global_momentum)
        to_host(delta_m)
        
        # record stat of m
        m_stats = compute_stats(debug(dct, sk, self.global_momentum, log_slots, weight_len, f"m{self.t}"))

        # compute delta_t for v using pre-normalization
        delta_v = [hn.Ciphertext(context) for _ in range(len(sum_real))]
        
        if self.args.server.algo == "original":
            v_denom = (1-beta2) ** 0.5
            mult(evt, self.global_v, beta2, self.global_v)
            
        if self.args.server.algo == "proposed":
            v_denom = ( (1-beta2)/(beta2**j_for_v) ) ** 0.5
            if j_for_v == 1 and self.t > 1:
                if (self.global_v[0].level == 3):
                    print("Bootstrapping v...")
                    bootstrap(bts, self.global_v)
                mult(evt, self.global_v, beta2**self.t2, self.global_v)
        
        v_denom /= C # for mean
        v_denom /= 2 # for adjusting the value generated in convert_to_real
        mult(evt, sum_real, v_denom, delta_v)
        if (delta_v[0].level == 3):
            print("Bootstrapping delta_v...")
            bootstrap(bts, delta_v)
        square(evt, delta_v, delta_v)
        
        # update v
        add(evt, self.global_v, delta_v, self.global_v)
        to_host(delta_v)
        
        # bootstrap v
        if self.global_v[0].level <= 8:
            print(f"Boostrapping v(level {self.global_v[0].level})...")
            bootstrap(bts, self.global_v)
        
        # record stat of v
        v_stats = compute_stats(debug(dct, sk, self.global_v, log_slots, weight_len, f"v{self.t}"))
        
        # record stat of delta
        delta_stats = compute_stats(debug(dct, sk, sum_real, log_slots, weight_len, f"delta{self.t}", 1/C/2))
        to_host(sum_real)
        
        # compute optimization step
        m_mult_lr = [hn.Ciphertext(context) for _ in range(len(self.global_momentum))]
        v_add_tau = [hn.Ciphertext(context) for _ in range(len(self.global_v))]

        mult(evt, self.global_momentum, server_lr, m_mult_lr) # mult lr
        add(evt, self.global_v, server_tau, v_add_tau) # add tau

        div2_v_add_tau = [hn.Ciphertext(context) for _ in range(len(v_add_tau))]
        mult(evt, v_add_tau, self.k*self.k/2, div2_v_add_tau) # (v+tau)/2
        scaled_v_add_tau = [hn.Ciphertext(context) for _ in range(len(v_add_tau))]
        mult(evt, v_add_tau, 2/(self.B-self.A), scaled_v_add_tau)
        sub(evt, scaled_v_add_tau, (self.B + self.A)/(self.B-self.A), scaled_v_add_tau) # scaled(v+tau)
        
        # invSqrt(v+tau)
        ret = HEInvSqrt(context, evt, bts, div2_v_add_tau, scaled_v_add_tau, 2**self.log_degree-1, self.iteration, self.A, self.B, self.k, weight_len)

        to_host(div2_v_add_tau)
        to_host(scaled_v_add_tau)

        ####### Debug invSqrt #######
        debug(dct, sk, ret, log_slots, weight_len, "invSqrt")
        v_add_tau_stats = compute_stats(debug(dct, sk, v_add_tau, log_slots, weight_len, "at the input"))
        to_host(v_add_tau)

        # compute opt step
        mult(evt, m_mult_lr, ret, ret)

        to_host(m_mult_lr)

        stats = {
            "delta": delta_stats,
            "m": m_stats,
            "v": v_stats,
            "v_add_tau": v_add_tau_stats
        }
        
        to_level(evt, ret, 0)

        ret_complex = [hn.Ciphertext(context) for _ in range((len(ret)+1)//2)]
        encrypted_real_to_complex(evt, ret, ret_complex)
        to_host(ret)

        return ret_complex, stats


@SERVER_REGISTRY.register()
class ServerAdam_PPRL(Server):    
    
    def __init__(self, args):
        super().__init__(args)

        self.t = 0

        # input domain for invSqrt
        self.A = args.server.A
        self.B = args.server.B

        # deg for invSqrt
        self.log_degree = args.server.log_degree
        self.k = 400
        
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + 1 * (10**-8)

        global_w = copy.deepcopy(model.state_dict())
        for key in global_w.keys():
            global_w[key] = torch.zeros_like(global_w[key]) + 1 / (1 * (10**-8) + self.args.server.tau)**0.5

        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v
        self.global_w = global_w
        
    def aggregate(self, local_deltas, client_ids, model_dict):
        C = len(client_ids)
        server_lr = self.args.trainer.global_lr
        
        for param_key in local_deltas:
            self.global_delta[param_key] = sum(local_deltas[param_key])/C
            self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
            self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1-self.args.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])
            self.global_w[param_key] = self.global_w[param_key] * ( (self.global_v[param_key]+self.args.server.tau) * self.global_w[param_key] * self.global_w[param_key] ) ** (-0.5)

        for param_key in model_dict.keys():
            model_dict[param_key] += server_lr * self.global_momentum[param_key] * self.global_w[param_key]
            
        delta_stats = compute_stats(flatten_to_numpy(self.global_delta))
        m_stats = compute_stats(flatten_to_numpy(self.global_momentum))
        v_stats = compute_stats(flatten_to_numpy(self.global_v))
        w_stats = compute_stats(flatten_to_numpy(self.global_w))
        
        stats = {
        "delta": delta_stats,
        "m": m_stats,
        "v": v_stats,
        "w": w_stats,
        }

        return model_dict, stats


    def encrypted_set_momentum(self, model):

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + 1 * (10**-8)

        global_w = copy.deepcopy(model.state_dict())
        for key in global_w.keys():
            global_w[key] = torch.zeros_like(global_v[key]) + 1 / ( (1 * (10**-8) + self.args.server.tau)**0.5 ) / self.k

        # heaan setting
        context, sk, pk, ect, dct, evt, bts, dt, log_slots = heaan_setting()
        num_slots = 2 ** log_slots

        # encrypt server's m, v, w
        global_momentum = flatten_to_numpy(global_momentum)
        global_v = flatten_to_numpy(global_v)
        global_w = flatten_to_numpy(global_w)

        # encryption
        encrypted_global_momentum = [hn.Ciphertext(context) for _ in range(math.ceil(len(global_momentum)/num_slots))]
        encrypted_global_v = [hn.Ciphertext(context) for _ in range(math.ceil(len(global_v)/num_slots))]
        encrypted_global_w = [hn.Ciphertext(context) for _ in range(math.ceil(len(global_v)/num_slots))]

        enc(ect, pk, dt, global_momentum, log_slots, encrypted_global_momentum)
        enc(ect, pk, dt, global_v, log_slots, encrypted_global_v)
        enc(ect, pk, dt, global_w, log_slots, encrypted_global_w)

        self.global_momentum = encrypted_global_momentum
        self.global_v = encrypted_global_v
        self.global_w = encrypted_global_w

    def encrypted_aggregate(self, encrypted_local_deltas, client_ids, weight_len):

        # index setting
        self.t += 1
        C = len(client_ids)

        beta1 = self.args.server.momentum
        beta2 = self.args.server.beta

        # update learning rate
        server_lr = copy.copy(self.args.trainer.global_lr)

        # update tau
        server_tau = copy.copy(self.args.server.tau)

        # heaan setting
        context, sk, pk, ect, dct, evt, bts, dt, log_slots = heaan_setting()
        
        # bootstrap m
        if self.global_momentum[0].level == 3:
            print("Bootstrapping m...")
            bootstrap(bts, self.global_momentum)
        # bootstrap v
        if self.global_v[0].level == 8:
            print("Bootstrapping v...")
            bootstrap(bts, self.global_v)
        # bootstrap w
        if self.global_w[0].level < 11:
            print(f"Bootstrapping w(level {self.global_w[0].level})...")
            bootstrap(bts, self.global_w)

        ####### Debug m_{t-1}, v_{t-1}, w_{t-1} #######
        # debug(dct, sk, self.global_momentum, log_slots, weight_len, f"m{self.t-1}")
        # debug(dct, sk, self.global_v, log_slots, weight_len, f"v{self.t-1}")
        # debug(dct, sk, self.global_w, log_slots, weight_len, f"w{self.t-1}", self.k)

        # compute sum of delta_t
        sum_complex = []
        for i in range(len(encrypted_local_deltas)):
            if i == 0:
                for j in range(len(encrypted_local_deltas[i])):
                    sum_complex.append(hn.Ciphertext(encrypted_local_deltas[i][j]))
                to_dt(sum_complex, dt)
            else:
                to_dt(encrypted_local_deltas[i], dt)
                add(evt, sum_complex, encrypted_local_deltas[i], sum_complex)
                to_host(encrypted_local_deltas[i])

            print(f"sum for client {i} success: num_ct", len(sum_complex))

        # bootstrap sum
        print(f"Bootstrapping sum(level {sum_complex[0].level})...")
        bootstrap(bts, sum_complex, is_complex=True)

        # convert sum to real value form
        sum_real = [hn.Ciphertext(context) for _ in range(len(self.global_momentum))]
        encrypted_complex_to_real(evt, sum_complex, sum_real, len(self.global_momentum))
        to_host(sum_complex)

        # compute delta_t for m using pre-normalization
        delta_m = [hn.Ciphertext(context) for _ in range(len(sum_real))]
        mult(evt, sum_real, (1-beta1)/C/2, delta_m)

        # update m
        mult(evt, self.global_momentum, beta1, self.global_momentum)
        add(evt, self.global_momentum, delta_m, self.global_momentum)
        to_host(delta_m)
        
        # record stat of m
        m_stats = compute_stats(debug(dct, sk, self.global_momentum, log_slots, weight_len, f"m{self.t}"))

        # compute delta_t for v using pre-normalization
        delta_v = [hn.Ciphertext(context) for _ in range(len(sum_real))]
        v_denom = (1-beta2)**0.5
        v_denom /= C # for mean
        v_denom /= 2 # for adjusting the value generated in convert_to_real
        mult(evt, sum_real, v_denom, delta_v)
        square(evt, delta_v, delta_v)

        # update v
        mult(evt, self.global_v, beta2, self.global_v)
        add(evt, self.global_v, delta_v, self.global_v)
        to_host(delta_v)

        # record stat of v
        v_stats = compute_stats(debug(dct, sk, self.global_v, log_slots, weight_len, f"v{self.t}"))
        
        # record stat of delta
        delta_stats = compute_stats(debug(dct, sk, sum_real, log_slots, weight_len, f"delta{self.t}", 1/C/2))
        to_host(sum_real)
        
        # compute optimization step
        # compute lr * m_t
        m_mult_lr = [hn.Ciphertext(context) for _ in range(len(self.global_momentum))]
        mult(evt, self.global_momentum, server_lr, m_mult_lr) # mult lr to m 
        
        # compute and scale (v_t + tau) * w_{t-1}^2
        v_add_tau = [hn.Ciphertext(context) for _ in range(len(self.global_v))]
        add(evt, self.global_v, server_tau, v_add_tau) # add tau to v
        
        v_mult_wSquared = [hn.Ciphertext(context) for _ in range(len(self.global_w))]
        integer_mult(evt, self.global_w, self.k, v_mult_wSquared) # w = w'*k
        square(evt, v_mult_wSquared, v_mult_wSquared) # w^2
        mult(evt, v_mult_wSquared, 2/(self.B-self.A), v_mult_wSquared) # 2*w^2/(B-A)
        mult(evt, v_add_tau, v_mult_wSquared, v_mult_wSquared) # 2*(v+tau)*w^2/(B-A)
        to_host(v_add_tau)
        
        sub(evt, v_mult_wSquared, (self.B+self.A)/(self.B-self.A), v_mult_wSquared) # 2*(v+tau)*w^2/(B-A) - (B+A)/(B-A)

        # compute invSqrt of (v_t + tau) * w_{t-1}^2
        ret = ChebysevInvSqrt_ct(evt, bts, v_mult_wSquared, 2**self.log_degree-1, self.A, self.B, 1)
        
        ####### Debug invSqrt #######
        debug(dct, sk, ret, log_slots, weight_len, "invSqrt")
        debug(dct, sk, v_mult_wSquared, log_slots, weight_len, "at the input")
        to_host(v_mult_wSquared)
        
        if ret[0].level == 3:
            print("Bootstrapping invSqrt result...")
            bootstrap(bts, ret)

        # compute w_t
        mult(evt, ret, self.global_w, self.global_w)
        w = [hn.Ciphertext(context) for _ in range(len(self.global_w))]
        integer_mult(evt, self.global_w, self.k, w)

        # record stat of W
        w_stats = compute_stats(debug(dct, sk, w, log_slots, weight_len, f"w{self.t}"))

        to_host(v_add_tau)

        # compute opt step
        mult(evt, m_mult_lr, w, ret)

        to_host(m_mult_lr)
        to_host(w)

        stats = {
            "delta": delta_stats,
            "m": m_stats,
            "v": v_stats,
            "w": w_stats,
        }

        to_level(evt, ret, 0)

        ret_complex = [hn.Ciphertext(context) for _ in range((len(ret)+1)//2)]
        encrypted_real_to_complex(evt, ret, ret_complex)
        to_host(ret)

        return ret_complex, stats


@SERVER_REGISTRY.register()
class ServerAdagrad(Server):    
    
    def __init__(self, args):
        super().__init__(args)

        self.t = 0

        self.t1 = args.server.t1

        # input domain for invSqrt
        self.A = args.server.A
        self.B = args.server.B

        # deg, #iter for invSqrt
        self.log_degree = args.server.log_degree
        self.iteration = args.server.iteration
        
        self.k = 1
        
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + 1 * (10**-8)

        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v
        
    def aggregate(self, local_deltas, client_ids, model_dict):
        
        C = len(client_ids)
        
        server_lr = self.args.trainer.global_lr
        server_tau = copy.copy(self.args.server.tau)
        
        if self.args.server.algo == "original":
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
                self.global_v[param_key] += self.global_delta[param_key] * self.global_delta[param_key]

        if self.args.server.algo == "proposed":
            self.t += 1

            i = 1 + (self.t - 1) % self.t1

            server_lr *= (self.args.server.momentum ** i)
            
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                
                if i == 1 and self.t > 1:
                    self.global_momentum[param_key] *= self.args.server.momentum ** self.t1

                self.global_momentum[param_key] += (1-self.args.server.momentum) / (self.args.server.momentum ** i) * self.global_delta[param_key]
                self.global_v[param_key] += self.global_delta[param_key] * self.global_delta[param_key]
            
        for param_key in model_dict.keys():
            if self.args.server.tau_in == True:
                model_dict[param_key] += server_lr * self.global_momentum[param_key] / ( (self.global_v[param_key] + server_tau) ** 0.5 )
            if self.args.server.tau_in == False:
                model_dict[param_key] += server_lr * self.global_momentum[param_key] / ( (self.global_v[param_key]**0.5)  + server_tau )
            
        delta_stats = compute_stats(flatten_to_numpy(self.global_delta))
        m_stats = compute_stats(flatten_to_numpy(self.global_momentum))
        v_stats = compute_stats(flatten_to_numpy(self.global_v))
        v_add_tau_stats = compute_stats(
            flatten_to_numpy({
                k: v + server_tau for k, v in self.global_v.items()
            })
        )
        
        stats = {
        "delta": delta_stats,
        "m": m_stats,
        "v": v_stats,
        "v_add_tau": v_add_tau_stats,
        }

        return model_dict, stats


    def encrypted_set_momentum(self, model):

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + 1 * (10**-8)

        # heaan setting
        context, sk, pk, ect, dct, evt, bts, dt, log_slots = heaan_setting()
        num_slots = 2 ** log_slots

        # encrypt server's m, v
        global_momentum = flatten_to_numpy(global_momentum)
        global_v = flatten_to_numpy(global_v)

        # encryption
        encrypted_global_momentum = [hn.Ciphertext(context) for _ in range(math.ceil(len(global_momentum)/num_slots))]
        encrypted_global_v = [hn.Ciphertext(context) for _ in range(math.ceil(len(global_v)/num_slots))]

        enc(ect, pk, dt, global_momentum, log_slots, encrypted_global_momentum)
        enc(ect, pk, dt, global_v, log_slots, encrypted_global_v)

        self.global_momentum = encrypted_global_momentum
        self.global_v = encrypted_global_v

    def encrypted_aggregate(self, encrypted_local_deltas, client_ids, weight_len):

        # index setting
        self.t += 1
        C = len(client_ids)

        i_for_m = 0
        if self.args.server.algo == "proposed":
            i_for_m = 1 + (self.t - 1) % self.t1

        beta1 = self.args.server.momentum

        # update learning rate
        server_lr = copy.copy(self.args.trainer.global_lr)
        server_lr *= (beta1 ** i_for_m)

        # update tau
        server_tau = copy.copy(self.args.server.tau)
        
        ####### Debug lr, tau #######
        print("t:", self.t)
        print("i:", i_for_m)
        print("beta1:", beta1)
        print("lr(before):", self.args.trainer.global_lr)
        print("tau(before):", self.args.server.tau)
        print("lr(after):", server_lr)
        print("tau(after):", server_tau)

        # heaan setting
        context, sk, pk, ect, dct, evt, bts, dt, log_slots = heaan_setting()

        # compute sum of delta_t
        sum_complex = []
        for i in range(len(encrypted_local_deltas)):
            if i == 0:
                for j in range(len(encrypted_local_deltas[i])):
                    sum_complex.append(hn.Ciphertext(encrypted_local_deltas[i][j]))
                to_dt(sum_complex, dt)
            else:
                to_dt(encrypted_local_deltas[i], dt)
                add(evt, sum_complex, encrypted_local_deltas[i], sum_complex)
                to_host(encrypted_local_deltas[i])
            print(f"sum for client {i} success: num_ct", len(sum_complex))

        # bootstrap sum
        if sum_complex[0].level <= 3:
            print(f"Bootstrapping sum(level {sum_complex[0].level})...")
            bootstrap(bts, sum_complex, is_complex=True)

        # convert sum to real value form
        sum_real = [hn.Ciphertext(context) for _ in range(len(self.global_momentum))]
        encrypted_complex_to_real(evt, sum_complex, sum_real, len(self.global_momentum))
        to_host(sum_complex)

        # compute delta_t for m using pre-normalization
        delta_m = [hn.Ciphertext(context) for _ in range(len(sum_real))]
        
        if self.args.server.algo == "original":
            mult(evt, sum_real, (1-beta1)/C/2, delta_m)
            if self.global_momentum[0].level == 3:
                print("Bootstrapping m...")
                bootstrap(bts, self.global_momentum)
            mult(evt, self.global_momentum, beta1, self.global_momentum)
            
        if self.args.server.algo == "proposed":
            mult(evt, sum_real, (1-beta1)/(beta1**i_for_m)/C/2, delta_m)
            if i_for_m == 1 and self.t > 1:
                if self.global_momentum[0].level == 3:
                    print("Bootstrapping m...")
                    bootstrap(bts, self.global_momentum)
                mult(evt, self.global_momentum, beta1**self.t1, self.global_momentum)

        # update m
        add(evt, self.global_momentum, delta_m, self.global_momentum)
        to_host(delta_m)
        
        # record stat of m
        m_stats = compute_stats(debug(dct, sk, self.global_momentum, log_slots, weight_len, f"m{self.t}"))

        # compute delta_t for v using pre-normalization
        delta_v = [hn.Ciphertext(context) for _ in range(len(sum_real))]
        
        v_denom = 1 / C # for mean
        v_denom /= 2 # for adjusting the value generated in convert_to_real
        mult(evt, sum_real, v_denom, delta_v)
        if (delta_v[0].level == 3):
            print("Bootstrapping delta_v...")
            bootstrap(bts, delta_v)
        square(evt, delta_v, delta_v)

        # update v
        add(evt, self.global_v, delta_v, self.global_v)
        to_host(delta_v)
        
        # bootstrap v
        if self.global_v[0].level <= 8:
            print(f"Boostrapping v(level {self.global_v[0].level})...")
            bootstrap(bts, self.global_v)
        
        # record stat of v
        v_stats = compute_stats(debug(dct, sk, self.global_v, log_slots, weight_len, f"v{self.t}"))
        
        # record stat of delta
        delta_stats = compute_stats(debug(dct, sk, sum_real, log_slots, weight_len, f"delta{self.t}", 1/C/2))
        to_host(sum_real)
        
        # compute optimization step
        m_mult_lr = [hn.Ciphertext(context) for _ in range(len(self.global_momentum))]
        v_add_tau = [hn.Ciphertext(context) for _ in range(len(self.global_v))]

        mult(evt, self.global_momentum, server_lr, m_mult_lr) # mult lr
        add(evt, self.global_v, server_tau, v_add_tau) # add tau

        div2_v_add_tau = [hn.Ciphertext(context) for _ in range(len(v_add_tau))]
        mult(evt, v_add_tau, self.k*self.k/2, div2_v_add_tau) # (v+tau)/2
        scaled_v_add_tau = [hn.Ciphertext(context) for _ in range(len(v_add_tau))]
        mult(evt, v_add_tau, 2/(self.B-self.A), scaled_v_add_tau)
        sub(evt, scaled_v_add_tau, (self.B + self.A)/(self.B-self.A), scaled_v_add_tau) # scaled(v+tau)

        # invSqrt(v+tau)
        ret = HEInvSqrt(context, evt, bts, div2_v_add_tau, scaled_v_add_tau, 2**self.log_degree-1, self.iteration, self.A, self.B, self.k, weight_len)

        to_host(div2_v_add_tau)
        to_host(scaled_v_add_tau)

        ####### Debug invSqrt #######
        debug(dct, sk, ret, log_slots, weight_len, "invSqrt")
        v_add_tau_stats = compute_stats(debug(dct, sk, v_add_tau, log_slots, weight_len, "at the input"))
        to_host(v_add_tau)

        # compute opt step
        mult(evt, m_mult_lr, ret, ret)

        to_host(m_mult_lr)

        stats = {
            "delta": delta_stats,
            "m": m_stats,
            "v": v_stats,
            "v_add_tau": v_add_tau_stats
        }

        to_level(evt, ret, 0)
        
        ret_complex = [hn.Ciphertext(context) for _ in range((len(ret)+1)//2)]
        encrypted_real_to_complex(evt, ret, ret_complex)
        to_host(ret)

        return ret_complex, stats