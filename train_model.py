import numpy as np
import tensorflow as tf

from data import get_data
from model import network2 as network


#do not specify an n value if you wish to use ALL mixed audio files
inputs, outputs = get_data()


for ind in range( len(inputs) ):
    inputs[ind] = np.reshape( inputs[ind], [1,160004,1] )
    inputs[ind] = inputs[ind].astype( 'float32' ) / 32768.0

for ind in range( len(outputs) ):
    outputs[ind] = np.reshape( outputs[ind], [1,160004,1] )
    outputs[ind] = outputs[ind].astype( 'float32' ) / 32768.0

if len(inputs) != len(outputs):
    raise('WARGH!')

tf.reset_default_graph()
sess = tf.Session()

learning_rate = 0.001
batch_size = 1
# sound_inwidth = 5120*8 # 16000 * 0.250  # 250 ms of context
# sound_outwidth = 160*8 # 16000 * 0.010  # 10ms of output
sound_inwidth = 160004
sound_outwidth = 160004

mixed_shape = [ batch_size, sound_inwidth, 1 ]
clean_shape = [ batch_size, sound_outwidth, 1 ]

mixed_audio = tf.placeholder( tf.float32, mixed_shape )
clean_audio = tf.placeholder( tf.float32, clean_shape )

with tf.name_scope( "model" ):
    denoised_audio = network( mixed_audio )

with tf.name_scope( "cost_function" ):
    # loss = tf.reduce_mean( tf.nn.l2_loss( denoised_audio - clean_audio ) )
    loss = tf.reduce_mean( 
        np.abs( 
            np.abs(  tf.fft3d( tf.cast( denoised_audio, tf.complex64 ) ) )**( 0.5 ) 
            - np.abs( tf.fft3d( tf.cast( clean_audio,tf.complex64 ) ) )**( 0.5 ) 
            ) 
        )

optim = tf.train.AdamOptimizer( learning_rate ).minimize( loss )

saver = tf.train.Saver()
# WARM START
#saver.restore( sess, "./model.ckpt" )
    
sess.run( tf.initialize_all_variables() )

# ================================================================

vals = []
best = np.inf
for iter in range( 1000000 ):


    """
    ind = np.random.choice( len(inputs) )
    ind2 = np.random.randint( low=0, high=160004-sound_inwidth-10 )

    # ind = 16
    #ind2 = 4200

    _, loss_val = sess.run( [optim,loss], feed_dict={mixed_audio:inputs[ind][0:1,ind2:ind2+sound_inwidth,0:1],
                                                     clean_audio:outputs[ind][0:1,ind2+sound_inwidth-sound_outwidth:ind2+sound_inwidth,0:1]} )
    """
    ind = np.random.choice( len(inputs) )
    _, loss_val = sess.run([optim, loss], feed_dict={mixed_audio:inputs[ind], clean_audio:outputs[ind]})
    
    vals.append( loss_val )

    if iter%10==0:
      np.save( 'vals.npy', vals )
      print("%d %.4f" % ( iter, loss_val ))
      if loss_val < best:
#          saver.save( sess, './model.ckpt' )
          best = loss_val



"""
outs = []
for ind2 in range( 0, 160004-sound_inwidth, sound_outwidth ):
    input = inputs[16][0:1,ind2:ind2+sound_inwidth,0:1]
    out_chunk = sess.run( denoised_audio, feed_dict={mixed_audio:input} )
    outs.append( np.reshape( out_chunk, [1,sound_outwidth] ))

output = np.hstack( outs )
# ouput =  np.reshape(inputs[16][0,0:155040,0],[1,155040])  + output
"""
input = inputs[16]
output = sess.run( denoised_audio, feed_dict={mixed_audio:input})


import scipy.io.wavfile as wf
wf.write( './output.wav', 16000, output.ravel() )

                                                   
