import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wf

# from data import get_data
from model import network2 as network


#do not specify an n value if you wish to use ALL mixed audio files
# inputs, outputs = get_data()


# for ind in range( len(inputs) ):
#     inputs[ind] = np.reshape( inputs[ind], [1,160004,1] )
#     inputs[ind] = inputs[ind].astype( 'float32' ) / 32768.0

# for ind in range( len(outputs) ):
#     outputs[ind] = np.reshape( outputs[ind], [1,160004,1] )
#     outputs[ind] = outputs[ind].astype( 'float32' ) / 32768.0

# if len(inputs) != len(outputs):
#     raise('WARGH!')

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
saver.restore( sess, "./model.ckpt" )

sr, data = wf.read("../noisy/testing.wav")

outs = []
for i in range(0,len(data)-160004,160004):
    data_in = data[i:i+160004,0]
    data_in = data_in.astype( 'float32' ) / 32768.0
    data_in = np.reshape( data_in, [1,160004,1] )
    out_chunk = sess.run(denoised_audio, feed_dict={mixed_audio:data_in})
    outs.append( np.reshape( out_chunk, [1, 160004]) )

output = np.hstack( outs )

wf.write("./cleaned_test.wav", 16000, output.ravel())