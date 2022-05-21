import sys
from time import time
import numpy as np
import logging
import net_opt_case as noc

from network_generator import gen_network
from base_network_opt import solve_network_model

logger = logging.getLogger("GA-NET-OP-LOGGER")
logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S'))
logger.addHandler(log_handler)



def main(customer_count, a_uv, e_u):
    result = solve_network_model(a_uv, customer_count, e_u, True)
    #print(",".join([str(i)for i in result[0]]))
    print(f"Case: customer_size={customer_count}, Objective is: {result[1]}")

if __name__ == '__main__':
    for case in noc.cases:
        main(case.indiv_size, case.a_uv, case.e_u)