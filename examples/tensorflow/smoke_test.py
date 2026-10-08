"""TensorFlow has the oldest native TPU support of any framework
(TPUStrategy). No compat layer needed, just the right cluster-resolver
boilerplate. This proves the TPU is visible and a basic op runs under it."""
import time

import tensorflow as tf


def main():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    print(f"tpu devices: {tf.config.list_logical_devices('TPU')}")

    with strategy.scope():
        a = tf.random.normal((2048, 2048))
        b = tf.random.normal((2048, 2048))

        @tf.function
        def matmul(a, b):
            return tf.matmul(a, b)

        start = time.time()
        c = strategy.run(matmul, args=(a, b))
        elapsed = time.time() - start

    assert c.shape == (2048, 2048)
    print(f"matmul ok, shape={c.shape}, first-compile+run={elapsed:.2f}s")


if __name__ == "__main__":
    main()
