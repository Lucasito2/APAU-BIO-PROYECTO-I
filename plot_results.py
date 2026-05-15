import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_data(num_seeds=5):
    """Carga los datos de los CSVs generados previamente"""
    seeds = [10, 42, 123, 777, 2024][:num_seeds]

    baseline_returns = []
    hybrid_returns = []
    hybrid_traces = []

    for seed in seeds:
        df_base = pd.read_csv(f'logs/baseline_seed_{seed}.csv')
        df_hybrid = pd.read_csv(f'logs/hybrid_stdp_seed_{seed}.csv')

        baseline_returns.append(df_base['return'].values)
        hybrid_returns.append(df_hybrid['return'].values)
        hybrid_traces.append(df_hybrid['mean_trace_magnitude'].values)

    return np.array(baseline_returns), np.array(hybrid_returns), np.array(hybrid_traces)

def smooth(data, window=20):
    """Aplica una media móvil para suavizar las curvas."""
    return np.array([np.convolve(row, np.ones(window)/window, mode='valid') for row in data])

def main():
    # Creamos directorio para las figuras
    os.makedirs('figures', exist_ok=True)

    # Cargamos los datos
    print("Cargando datos de los registros...")
    try:
        base_ret, hybrid_ret, hybrid_traces = load_data()
    except FileNotFoundError:
        print("ERROR: No se encuentran los CSVs. Ejecuta primero run_full_experiment().")
        return

    # Calculamos Estadísticos Oficiales (Parte I del Boletín)
    #  Y puntuación de convergencia (media del 10% final)
    k_final = int(base_ret.shape[1] * 0.1)
    base_conv = np.mean(base_ret[:, -k_final:], axis=1) # Medias por semilla
    hybrid_conv = np.mean(hybrid_ret[:, -k_final:], axis=1)

    mean_base = np.mean(base_conv)
    mean_hybrid = np.mean(hybrid_conv)

    # Mejora Relativa Delta
    delta = (mean_hybrid - mean_base) / abs(mean_base)

    # Prueba t de Welch
    t_stat, p_value = stats.ttest_ind(hybrid_conv, base_conv, equal_var=False)

    print("\n====================================================================")
    print(" RESULTADOS ESTADÍSTICOS (TABLA II del LaTeX)")
    print("====================================================================")
    print(f"Convergencia Línea Base (Adam):  {mean_base:.2f} ± {np.std(base_conv):.2f}")
    print(f"Convergencia Híbrido (STDP):     {mean_hybrid:.2f} ± {np.std(hybrid_conv):.2f}")
    print(f"Mejora Relativa (Delta):         {delta * 100:.2f}%")
    print(f"Valor-p (Prueba Welch):          {p_value:.5f}")
    print("--------------------------------------------------------------------")
    if p_value < 0.05:
        print("-> La diferencia es ESTADÍSTICAMENTE SIGNIFICATIVA (p < 0.05)")
    else:
        print("-> La diferencia NO es estadísticamente significativa (p >= 0.05)")
    print("====================================================================\n")

    # Generamos las Figuras
    print("Generando y guardando figuras...")

    # Figura 1: Curvas de Aprendizaje (con banda de varianza)
    plt.figure(figsize=(10, 6))

    # Suavizamos y calculamos medias y desviaciones estándar
    base_smooth = smooth(base_ret)
    hyb_smooth = smooth(hybrid_ret)

    x_axis = range(19, base_ret.shape[1]) # Eje ajustado por la ventana de suavizado

    # Plot Baseline
    plt.plot(x_axis, np.mean(base_smooth, axis=0), color='gray', label='Línea Base (Adam)')
    plt.fill_between(x_axis,
                     np.mean(base_smooth, axis=0) - np.std(base_smooth, axis=0),
                     np.mean(base_smooth, axis=0) + np.std(base_smooth, axis=0),
                     color='gray', alpha=0.2)

    # Plot Hybrid
    plt.plot(x_axis, np.mean(hyb_smooth, axis=0), color='blue', label='Híbrido (STDP)')
    plt.fill_between(x_axis,
                     np.mean(hyb_smooth, axis=0) - np.std(hyb_smooth, axis=0),
                     np.mean(hyb_smooth, axis=0) + np.std(hyb_smooth, axis=0),
                     color='blue', alpha=0.2)

    plt.title('Curvas de Aprendizaje: STDP vs Adam')
    plt.xlabel('Episodios')
    plt.ylabel('Retorno de Episodio (Media Móvil)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('figures/learning_curves.png', dpi=300)
    plt.close()

    # Figura 2: Dinámica Interna (Diagnóstico Obligatorio)
    plt.figure(figsize=(10, 4))
    traces_mean = np.mean(hybrid_traces, axis=0)
    traces_std = np.std(hybrid_traces, axis=0)

    plt.plot(traces_mean, color='darkorange', label='Magnitud Media de Trazas e_ij')
    plt.fill_between(range(len(traces_mean)),
                     traces_mean - traces_std,
                     traces_mean + traces_std,
                     color='darkorange', alpha=0.2)

    plt.title('Dinámica Interna: Evolución de las Trazas STDP')
    plt.xlabel('Episodios')
    plt.ylabel('Magnitud Absoluta')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('figures/internal_dynamics.png', dpi=300)
    plt.close()

    print("Figuras guardadas correctamente en la carpeta 'figures/'.")

if __name__ == '__main__':
    main()
