import numpy as np
def sigmoid(x):
    return np.power(1+np.exp(-x), -1)

def dsigmoid(x):
    t=sigmoid(x)
    return (1-t)*t

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return  1-np.square(np.tanh(x))

def softmax(x):
    xexp = np.exp(x)
    esum = np.sum(xexp)
    return xexp/esum

import pickle

class gru:

    def __init__(self, i_size, h_size, o_size, optimize='rmsprop', wb=None):

        self.i_size = i_size
        self.h_size = h_size
        self.o_size = o_size
        self.optimize = optimize

        # self.names = {'ur':0,'wr':1, 'uz':2, 'wz':3, 'u_h':4, 'w_h':5, 'wo':6}
        self.names = {0:'ur',1:'wr', 2:'uz', 3:'wz', 4:'u_h', 5:'w_h', 6:'wo'}
        if wb:
            self.w, self.b = self.load_weights(wb)

        else:
            self.w={}
            self.b={}

            # reset weights
            self.w['ur'] = np.random.normal(0,0.01,(h_size, i_size))
            self.b['r'] = np.zeros((h_size, 1))
            self.w['wr'] = np.random.normal(0,0.01,(h_size, h_size))

            # update weights
            self.w['uz'] = np.random.normal(0,0.01,(h_size, i_size))
            self.b['z'] = np.zeros((h_size, 1))
            self.w['wz'] = np.random.normal(0,0.01,(h_size, h_size))

            # _h weights
            self.w['u_h'] = np.random.normal(0,0.01,(h_size, i_size))
            self.b['_h'] = np.zeros((h_size, 1))
            self.w['w_h'] = np.random.normal(0,0.01,(h_size, h_size))

            # out weight
            self.w['wo'] = np.random.normal(0,0.01,(o_size, h_size))
            self.b['o'] = np.zeros((o_size, 1))

        if optimize == 'rmsprop' or optimize == 'adam':
            self.m={}
            self.m['ur'] = np.zeros((h_size, i_size))
            self.m['wr'] = np.zeros((h_size, h_size))
            self.m['uz'] = np.zeros((h_size, i_size))
            self.m['wz'] = np.zeros((h_size, h_size))
            self.m['u_h'] = np.zeros((h_size, i_size))
            self.m['w_h'] = np.zeros((h_size, h_size))
            self.m['wo'] = np.zeros((o_size, h_size))

        if optimize == 'adam':
            self.v={}
            self.v['ur'] = np.zeros((h_size, i_size))
            self.v['wr'] = np.zeros((h_size, h_size))
            self.v['uz'] = np.zeros((h_size, i_size))
            self.v['wz'] = np.zeros((h_size, h_size))
            self.v['u_h'] = np.zeros((h_size, i_size))
            self.v['w_h'] = np.zeros((h_size, h_size))
            self.v['wo'] = np.zeros((o_size, h_size))
            self.weight_update = adam

        elif optimize == 'rmsprop':
            self.weight_update = rmsprop

    def forward_pass(self, inputs):

        self.inputs = inputs
        self.n_inp = len(inputs)
        self.vr = []; self.vz = []; self.v_h = []; self.vo = [];
        self.r=[]; self.z=[]; self._h=[]; self.h={}; self.o = []
        self.h[-1] = np.zeros((self.h_size,1))
        for i in range(self.n_inp):

            # calculating reset gate value
            self.vr.append(np.dot(self.w['ur'],inputs[i]) + np.dot(self.w['wr'], self.h[i-1]) + self.b['r'])
            self.r.append(sigmoid(self.vr[i]))

            # calculation update gate value
            self.vz.append(np.dot(self.w['uz'],inputs[i]) + np.dot(self.w['wz'], self.h[i-1])  + self.b['z'])
            self.z.append(sigmoid(self.vz[i]))

            # applying reset gate value
            self.v_h.append(np.dot(self.w['u_h'], inputs[i]) + np.dot(self.w['w_h'], np.multiply(self.h[i - 1], self.r[i])) +  + self.b['_h'])
            self._h.append(tanh(self.v_h[i]))

            # applying update gate value
            self.h[i] = np.multiply(self.z[i], self.h[i - 1]) + np.multiply(1-self.z[i], self._h[i])

            # # calculating output
            # self.vo.append(np.dot(self.w['wo'], self.h[i]))
            # self.o.append(sigmoid(self.vo[i]))

            # calculating output
            self.vo.append(np.dot(self.w['wo'], self.h[i]) + self.b['o'])
            self.o.append(softmax(self.vo[i]))
        return self.o

    def backward_pass(self, t):
        # error calculation
        e = self.error(t)

        # dw variables
        dw={}
        db= {}
        dw['uz'] = np.zeros((self.h_size, self.i_size))
        db['z'] = np.zeros((self.h_size, 1))
        dw['wz'] = np.zeros((self.h_size, self.h_size))

        # reset dw
        dw['ur'] = np.zeros((self.h_size, self.i_size))
        db['r'] = np.zeros((self.h_size, 1))
        dw['wr'] = np.zeros((self.h_size, self.h_size))

        # _h dw
        dw['u_h'] = np.zeros((self.h_size, self.i_size))
        db['_h'] = np.zeros((self.h_size, 1))
        dw['w_h'] = np.zeros((self.h_size, self.h_size))

        # hidden-2-output dw
        dw['wo'] = np.zeros((self.o_size, self.h_size))
        db['o'] = np.zeros((self.o_size, 1))

        dh = 0.0

        for i in reversed(range(self.n_inp)):

            # gradient at output layer
            go = self.o[i] - t[i]

            # hidden to outpur weight's dw
            dw['wo'] += np.dot(go, self.h[i].T)
            db['o'] += go

            # gradient at top hidden layer
            dh += np.dot(self.w['wo'].T, go)

            dz = (self.h[i-1] - self._h[i]) * dh
            dz__ = self.z[i] * dh
            # dz_ = dsigmoid(self.vz[i]) * dz
            dz_ = np.multiply((1.0 - self.z[i]), self.z[i]) * dz

            d_h = (1-self.z[i]) * dh
            d_h_ = (1- np.square(self._h[i])) * d_h

            temp = np.dot(self.w['w_h'].T, d_h_)
            dr = self.h[i-1] * temp
            dr_ = np.multiply((1.0 - self.r[i]), self.r[i]) * dr
            dr__ = self.r[i] * temp

            # calculating reset dw
            dw['ur'] += np.dot(dr_ , self.inputs[i].T)
            db['r'] += dr_
            dw['wr'] += np.dot(dr_ , self.h[i-1].T)
            # db['wr'] += dr_

            # calculating update dw
            dw['uz'] += np.dot(dz_, self.inputs[i].T)
            db['z'] += dz_
            dw['wz'] += np.dot(dz_, self.h[i-1].T)
            # db['wz'] += dz_

            # calculating _h dw
            dw['u_h'] += np.dot(d_h_, self.inputs[i].T)
            db['_h'] += d_h_
            dw['w_h'] += np.dot(d_h_, np.multiply(self.r[i], self.h[i-1]).T)
            # db['w_h'] += d_h_

            dh = np.dot(self.w['wr'].T, dr_) + np.dot(self.w['wz'].T, dz_) + dz__ + dr__

        return dw, db, np.linalg.norm(e)

    def error(self, t):
        loss = np.sum(t * np.log(self.o))
        return -loss

    def save_model(self, fname):
        pickle.dump([self.w, self.b], open(fname, 'wb'))

    def load_model(self, fname):
        return pickle.load(open(fname, 'rb'))

            # def reset_m(self):
    #     if self.optimize == 'rmsprop' or self.optimize == 'adam':
    #         self.m = {}
    #         self.m['ur'] = np.zeros((self.h_size, self.i_size))
    #         self.m['wr'] = np.zeros((self.h_size, self.h_size))
    #         self.m['uz'] = np.zeros((self.h_size, self.i_size))
    #         self.m['wz'] = np.zeros((self.h_size, self.h_size))
    #         self.m['u_h'] = np.zeros((self.h_size, self.i_size))
    #         self.m['w_h'] = np.zeros((self.h_size, self.h_size))
    #         self.m['wo'] = np.zeros((self.o_size, self.h_size))
    #
    #     if self.optimize == 'adam':
    #         self.v = {}
    #         self.v['ur'] = np.zeros((self.h_size, self.i_size))
    #         self.v['wr'] = np.zeros((self.h_size, self.h_size))
    #         self.v['uz'] = np.zeros((self.h_size, self.i_size))
    #         self.v['wz'] = np.zeros((self.h_size, self.h_size))
    #         self.v['u_h'] = np.zeros((self.h_size, self.i_size))
    #         self.v['w_h'] = np.zeros((self.h_size, self.h_size))
    #         self.v['wo'] = np.zeros((self.o_size, self.h_size))
    #     return


def rmsprop(self, dw, db, neta, b1=.9, b2=.0, e=1e-8):
        for wpi, g in dw.items():
            self.m[wpi] = b1 * self.m[wpi] + (1 - b1) * np.square(g)
            self.w[wpi] -= neta * np.divide(g, (np.sqrt(self.m[wpi]) + e))
        for wpi in db:
            self.b[wpi] -= neta * db[wpi]
        return

def adam(self, dw, db, neta, b1=0.9, b2=0.99, e=1e-8):
    for wpi, g in dw.items():
        self.m[wpi] = (b1 * self.m[wpi]) + ((1. - b1) * g)

        self.v[wpi] = (b2 * self.v[wpi]) + ((1. - b2) * np.square(g))

        m_h = self.m[wpi]/(1.-b1)
        v_h = self.v[wpi]/(1.-b2)

        # w[wpi] -= neta * (m_h/(np.sqrt(v_h) + e) + regu * w[wpi])
        self.w[wpi] -= neta * m_h/(np.sqrt(v_h) + e)
    for wpi in db:
        self.b[wpi] -= neta * db[wpi]
    return