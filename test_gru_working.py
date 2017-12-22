import numpy as np, pickle
from GRU import gru
from random import shuffle

def sample_formation(text, seq_length, map_vect):
    samples = []
    t_size = len(text)
    for i in range(0, t_size - seq_length - 1):
        x = [map_vect[j] for j in text[i: i + seq_length]]
        y = [map_vect[j] for j in text[i + 1: i + seq_length + 1]]
        samples.append((x, y))
    return samples

if __name__=='__main__':
    text = open('pg.txt','r').read()
    chars = list(set(text))
    v_size, t_size = len(chars), len(text)

    # vector formation for all character
    map_vect={}
    for i in range(len(chars)):
        map_vect[chars[i]] = np.zeros((v_size,1))
        map_vect[chars[i]][i] = 1.0

    # recurrent NN initalization
    rcc_layer = gru(v_size, 250, v_size, optimize='rmsprop')

    # sample generation
    seq_length = 25
    samples = sample_formation(text, seq_length, map_vect)

    # RNN training parameter
    batch = 100
    miter = 20
    epoch = 50

    print "training start."
    while epoch > 0:
        itr = 0
        while itr < miter:
            deltaw = {'ur':0.0,'wr':0.0, 'uz':0.0, 'wz':0.0, 'u_h':0.0, 'w_h':0.0, 'wo':0.0}
            deltab= {'r':0.0, 'z':0.0, '_h':0.0, 'o':0.0}
            err = 0

            # mini_batch foramtion
            mini_batch = [samples[np.random.randint(0, len(samples))] for i in range(batch)]

            # mini_batch training
            while mini_batch:
                x,y = mini_batch.pop()
                rcc_layer.forward_pass(x)
                dw, db, e = rcc_layer.backward_pass(y)
                for j in dw:
                    deltaw[j] += dw[j]
                for j in db:
                    deltab[j]+=db[j]
                err += e

            # updating Recurrent network
            rcc_layer.weight_update(rcc_layer, {j:deltaw[j]/batch for j in deltaw}, {j:deltab[j]/batch for j in deltab}, neta=0.01)
            print '\t',itr,"batch error is",err/batch
            itr += 1

        print "\n %d epoch is completed" % (epoch)
        epoch -= 1
    print "training complete."
    rcc_layer.save_model('weights.pickle')

    # setting testing parameters
    iters = 1000
    correct = 0.0
    itr = 0

    # testing of RNN
    print "\ntesting start."
    while itr < iters:

        # selecting random sample from samples
        x, y = samples[np.random.randint(0, len(samples))]
        _o = rcc_layer.forward_pass(x)
        # print np.argmax(_o[-1]), np.argmax(y[-1])
        if np.argmax(_o[-1]) == np.argmax(y[-1]):
            correct += 1
        itr += 1
    print "\ntesting complete."

    print "correct:\t",correct
    print "incorrect:\t",iters-correct

    print "\naccuracy:\t",correct/iters
