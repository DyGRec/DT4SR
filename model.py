from modules import *


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        src_masks = tf.math.equal(self.input_seq, 0)

        with tf.variable_scope("mean_SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.mean_seq, item_mean_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_mean_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Positional Encoding
            mean_t, pos_mean_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_mean_pos",
                reuse=reuse,
                with_t=True
            )
            self.mean_seq += mean_t

            # Dropout
            self.mean_seq = tf.layers.dropout(self.mean_seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.mean_seq *= mask

            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_mean_blocks_%d" % i):

                    # Self-attention
                    self.mean_seq = multihead_attention(queries=normalize(self.mean_seq),
                                                   keys=self.mean_seq,
                                                   values=self.mean_seq,
                                                   key_masks=src_masks,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   training=self.is_training,
                                                   causality=True,
                                                   scope="mean_self_attention")

                    # Feed forward
                    self.mean_seq = feedforward(self.mean_seq, num_units=[args.hidden_units, args.hidden_units])
                    self.mean_seq *= mask

            self.mean_seq = normalize(self.mean_seq)
        

        with tf.variable_scope("var_SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.var_seq, self.item_var_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_var_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            # Positional Encoding
            var_t, pos_var_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_var_pos",
                reuse=reuse,
                with_t=True
            )
            self.var_seq += var_t

            # Dropout
            self.var_seq = tf.layers.dropout(self.var_seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.var_seq *= mask

            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("var_mean_blocks_%d" % i):

                    # Self-attention
                    self.var_seq = multihead_attention(queries=normalize(self.var_seq),
                                                   keys=self.var_seq,
                                                   values=self.var_seq,
                                                   key_masks=src_masks,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   training=self.is_training,
                                                   causality=True,
                                                   scope="var_self_attention")

                    # Feed forward
                    self.var_seq = feedforward(self.var_seq, num_units=[args.hidden_units, args.hidden_units])
                    self.var_seq *= mask

            self.var_seq = tf.nn.elu(normalize(self.var_seq)) + 1



        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_mean_emb = tf.nn.embedding_lookup(item_mean_emb_table, pos)
        pos_var_emb = tf.nn.elu(tf.nn.embedding_lookup(self.item_var_emb_table, pos)) + 1
        neg_mean_emb = tf.nn.embedding_lookup(item_mean_emb_table, neg)
        neg_var_emb = tf.nn.elu(tf.nn.embedding_lookup(self.item_var_emb_table, neg)) + 1
        seq_mean_emb = tf.reshape(self.mean_seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])
        seq_var_emb = tf.reshape(self.var_seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(args.evalnegsample+1))
        #test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        test_item_mean_emb = tf.nn.embedding_lookup(item_mean_emb_table, self.test_item)
        test_item_var_emb = tf.nn.elu(tf.nn.embedding_lookup(self.item_var_emb_table, self.test_item)) + 1

        test_user_mean_emb = self.mean_seq[:, -1, :]
        test_user_mean_emb = tf.tile(tf.expand_dims(test_user_mean_emb, 1), [1, tf.shape(test_item_mean_emb)[0], 1])
        test_user_mean_emb = tf.reshape(test_user_mean_emb, [tf.shape(self.input_seq)[0]*tf.shape(test_item_mean_emb)[0], args.hidden_units])
        test_user_var_emb = self.var_seq[:, -1, :]
        test_user_var_emb = tf.tile(tf.expand_dims(test_user_var_emb, 1), [1, tf.shape(test_item_mean_emb)[0], 1])
        test_user_var_emb = tf.reshape(test_user_var_emb, [tf.shape(self.input_seq)[0]*tf.shape(test_item_mean_emb)[0], args.hidden_units])
        #self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        #pred_data = (seq_mean_emb, seq_var_emb, test_item_mean_emb, test_item_var_emb)
        #self.test_logits = tf.map_fn(lambda x: wasserstein(x[0], x[1], x[2], x[3]), pred_data, dtype=tf.float32, parallel_iterations=20)
        self.test_logits = wasserstein(test_user_mean_emb, test_user_var_emb, test_item_mean_emb, test_item_var_emb)
        #self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, itemnum+1])
        #self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        #self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        #self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)
        #pred_pos_data = (seq_mean_emb, seq_var_emb, pos_mean_emb, pos_var_emb)
        #pred_neg_data = (seq_mean_emb, seq_var_emb, neg_mean_emb, neg_var_emb)
        #self.pos_logits = tf.map_fn(lambda x: wasserstein(x[0], x[1], x[2], x[3]), pred_pos_data, dtype=tf.float32, parallel_iterations=20)
        #self.neg_logits = tf.map_fn(lambda x: wasserstein(x[0], x[1], x[2], x[3]), pred_pos_data, dtype=tf.float32, parallel_iterations=20)

        self.pos_logits = wasserstein(seq_mean_emb, seq_var_emb, pos_mean_emb, pos_var_emb)
        self.neg_logits = wasserstein(seq_mean_emb, seq_var_emb, neg_mean_emb, neg_var_emb)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        #self.loss = tf.reduce_sum(
        #    - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
        #    tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        #) / tf.reduce_sum(istarget)
        self.loss = tf.reduce_sum(
                -tf.log(tf.sigmoid(self.neg_logits - self.pos_logits + 1e-24)) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += args.l2_emb * sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.neg_logits - self.pos_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, test_items):
        return sess.run(self.test_logits,
                {self.u: u, self.input_seq: seq, self.test_item: test_items, self.is_training: False})
    
def KL(mean1, cov1, mean2, cov2):
    mat_cov1_inv = tf.diag(tf.reciprocal(cov1))
    mat_cov1 = tf.diag(cov1)
    mat_cov2 = tf.diag(cov2)
    ret = tf.trace(tf.matmul(mat_cov1_inv, mat_cov2))

    diff = tf.reshape(mean1 - mean2, (config.shape[-1], 1))
    ret = ret + tf.matmul(tf.matmul(tf.transpose(diff), mat_cov1_inv), diff)
    ret = ret - tf.log(tf.divide(tf.math.reduce_prod(cov2), tf.math.reduce_prod(cov1)))
    ret = ret - config.shape[-1]
    return ret/2


def wasserstein(mean1, cov1, mean2, cov2):
    ret = tf.reduce_sum((mean1 - mean2) * (mean1 - mean2), axis=1)
    temp = tf.sqrt(tf.maximum(cov1, 1e-9)) - tf.sqrt(tf.maximum(cov2, 1e-9))
    ret = ret + tf.reduce_sum(temp * temp, axis=1)
    #mat_cov1 = tf.diag(cov1)
    #mat_cov2 = tf.diag(cov2)
    #trace = mat_cov1 + mat_cov2 - tf.diag(2*tf.math.sqrt((tf.math.sqrt(cov2) * cov1 * tf.math.sqrt(cov2))))
    #trace = tf.trace(trace)
    #return ret + trace
    return ret
