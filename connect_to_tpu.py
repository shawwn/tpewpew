import sys
from pprint import pprint as pp

import tensorflow as tf2

tf = tf2.compat.v1


def main():
  master, *args = sys.argv[1:]
  job_name = "tpu_worker"
  if len(args) > 0:
    job_name, *args = args[0], args[1:]
    
  tf.config.experimental_connect_to_host(master, job_name=job_name)
  pp(tf.config.list_logical_devices())


if __name__ == '__main__':
  main()

