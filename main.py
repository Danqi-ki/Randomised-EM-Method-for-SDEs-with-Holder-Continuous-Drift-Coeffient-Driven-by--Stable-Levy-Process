import numpy as np
import sys
from configparser import ConfigParser
import os
import matplotlib


from src.RandomisedEM import (
    LevyBase,
    ref_sols,
    reduced_randomized_em_sde_error,
    reduced_em_sde_error,
    error_plot_allinone
)
from src.drift_example import (drift_weierstrass, drift_jump, drift_root, drift_time_oscillation)
matplotlib.use('Agg')  


def LGen(T,time_factors_list,fine_sol_time_factor, n_experiments,alpha, beta, mu, base_number):

    #finest level Levy generated
    n_list= base_number**time_factors_list
    N=n_list[-1]*fine_sol_time_factor

    return LevyBase(T,n_experiments,N, alpha=alpha, beta=beta, mu=mu)


def error_plots(T, time_factors_list,
                fine_sol_time_factor,
                n_experiments,
                L,
                f,args_f,
                base_number=2, initial=1.1,
                savefig=False, reuler_only=False,
                alpha=None):

    n_list= base_number**time_factors_list


    #finest level  
    N=n_list[-1]*fine_sol_time_factor

    ###to get the reference solution:
    print("Sol Ref Begin!")
    SOL_REF=ref_sols(T,  L, N, n_experiments,  f, args_f,initial=initial)

    print("Sol Ref DONE!")

    print("Randomised Euler-Maruyama Begin!")
    error_list_random=reduced_randomized_em_sde_error(SOL_REF,T, L, n_list,
                                                  fine_sol_time_factor,
                                                  n_experiments,
                                                  f,
                                                  args_f,
                                                  initial=initial,
                                                  base_number=base_number)
    print("Randomised Euler-Maruyama DONE!")


    print("Euler Begin!")
    error_list_de=reduced_em_sde_error(SOL_REF,T, L, n_list,
                                      fine_sol_time_factor,
                                      n_experiments,
                                      f,args_f,
                                      initial=initial,
                                      base_number=base_number)
    print("Euler DONE!")

    if savefig:
        data_name = savefig[:-4]
        np.savez(data_name,
                 errVec=error_list_random,
                 errVecDe=error_list_de,
                 timeFac=time_factors_list,
                 alpha=alpha)


    error_plot_allinone(base_number, [error_list_random, error_list_de],
                            time_factors_list,
                            savefig=savefig)
    return SOL_REF



os.makedirs("errorplots", exist_ok=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_path = os.path.join("config", sys.argv[1])
    config = { "np": np,
    "os": os,
    "sys": sys}
    with open(config_path, "r") as f:
        exec(f.read(), config)

    drift_type = config_path.split("/")[-1].split(".")[0].split("_")[1] 
    
    f_map = {
        "weier": (drift_weierstrass, ("a", "b", "n_terms", "m_terms")),
        "jump": (drift_jump, ("T",)),
        "root": (drift_root, ("T",)),
        "time": (drift_time_oscillation, ("lbd", "order1", "order2", "w")),
    }

    f, param_names = f_map[drift_type]
    args_f = tuple(config[name] for name in param_names) if param_names else ()

    # Unpack shared parameters
    n_experiments = config["n_experiments"]
    T = config["T"]
    base_number = config["base_number"]
    time_factors_list = config["time_factors_list"]
    fine_sol_time_factor = config["fine_sol_time_factor"]
    alpha = config["alpha"]
    beta = config["beta"]
    mu = config["mu"]
    initial = config["initial"]
    savefig = config["savefig"]

    # Generate Lévy noise
    L = LGen(T, time_factors_list, fine_sol_time_factor, n_experiments, alpha, beta, mu, base_number)



    # Run experiment
    sol_ref = error_plots(T, time_factors_list, fine_sol_time_factor, n_experiments, L, f, args_f,
                          base_number=base_number, initial=initial, savefig=savefig, alpha=alpha)

    print(f"\n✅ Experiment complete. Results saved in: {savefig}")



