from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Konštanty z merania piezoaktuátora
LAMBDA = 532.0  # nm (vlnová dĺžka) pouziteho svetla
K_PIEZO = 100.817  # nm/V (citlivosť posunu aktuátora) --> kalibracia pieza

def model_funkcia(
    u_piezo, u0=0.0, a=1.0, phi0=0.0, k_piezo=K_PIEZO, lmbda=LAMBDA
):
    """Vypočíta teoretické napätie U(t) podľa vzorca:

    U(t) = U0 + A * cos^2( 2*phi0 + (2*pi / lambda) * k_piezo * Upiezo(t) )
    """
    faza = 2 * phi0 + (2 * np.pi / lmbda) * k_piezo * u_piezo
    return u0 + a * (np.cos(faza) ** 2)


def nacitaj_a_vykresli_osciloskop_s_fitom(
    priecinok_path,
    show_ch1=True,
    scale_ch1=False,
    show_ch2=True,
    show_model=True,
):
    cesta = Path(priecinok_path)

    if not cesta.exists():
        print(f"Chyba: Cesta '{priecinok_path}' neexistuje.")
        return

    csv_subory = [cesta] if cesta.is_file() else sorted(list(cesta.glob("*.csv")))

    if not csv_subory:
        print("Nenašli sa žiadne CSV súbory.")
        return

    plt.figure(figsize=(11, 6))

    for subor in csv_subory:
        try:
            # Načítanie dát z osciloskopu (preskočíme 2 riadky hlavičky)
            df = pd.read_csv(
                subor,
                skiprows=2,
                header=None,
                names=["Time", "CH1", "CH2"],
                skipinitialspace=True,
            )
            df = df.apply(pd.to_numeric, errors="coerce").dropna()

            u_piezo = df["CH1"].values
            u_namerane = df["CH2"].values
            cas_ms = df["Time"].values * 1e3

            # Vykreslenie Kanála 1 (Upiezo) - obyčajné alebo naskálované
            if show_ch1 and "CH1" in df.columns:
                if scale_ch1 and "CH2" in df.columns:
                    # Naskálovanie CH1 na amplitúdu a stred CH2
                    ptp_ch1 = np.ptp(u_piezo)
                    ptp_ch2 = np.ptp(u_namerane)
                    scale_factor = (
                        ptp_ch2 / ptp_ch1 if ptp_ch1 != 0 else 1.0
                    )
                    u_ch1_plot = (u_piezo - np.mean(u_piezo)) * scale_factor + np.mean(
                        u_namerane
                    )
                    label_ch1 = f"{subor.stem} - CH1 Naskálovaný ($\times{scale_factor:.2e}$)"
                else:
                    u_ch1_plot = u_piezo
                    label_ch1 = f"{subor.stem} - CH1 ($U_{{piezo}}$)"

                plt.plot(
                    cas_ms, u_ch1_plot, label=label_ch1, linewidth=1.2, alpha=0.6
                )

            # Vykreslenie Kanála 2 (Nameraná odozva U)
            if show_ch2 and "CH2" in df.columns:
                plt.plot(
                    cas_ms,
                    u_namerane,
                    label=f"{subor.stem} - CH2 (Namerané U)",
                    linewidth=1.2,
                    alpha=0.7,
                )

            # Automatický fit a vykreslenie teoretického modelu
            if show_model and "CH1" in df.columns and "CH2" in df.columns:
                u0_est = np.min(u_namerane)
                a_est = np.ptp(u_namerane)

                # Obmedzenia pre fitovanie
                bounds = (
                    [u0_est * 0.9, a_est * 0.8, -np.pi],
                    [u0_est * 1.1, a_est * 1.2, np.pi],
                )

                popt, pcov = curve_fit(
                    lambda x, u0, a, phi0: model_funkcia(
                        x, u0, a, phi0, k_piezo=K_PIEZO, lmbda=LAMBDA
                    ),
                    u_piezo,
                    u_namerane,
                    p0=[u0_est, a_est, 0.0],
                    bounds=bounds,
                )

                perr = np.sqrt(np.diag(pcov))
                residuals = u_namerane - model_funkcia(
                    u_piezo, *popt, k_piezo=K_PIEZO, lmbda=LAMBDA
                )
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((u_namerane - np.mean(u_namerane)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                print(f"=== Výsledky fittu pre {subor.name} ===")
                print(f"  u0   = {popt[0]:.6f} ± {perr[0]:.6f} V")
                print(f"  a    = {popt[1]:.6f} ± {perr[1]:.6f} V")
                print(f"  phi0 = {popt[2]:.6f} ± {perr[2]:.6f} rad")
                print(f"  R²   = {r_squared:.4f}\n")

                u_teoria_fit = model_funkcia(
                    u_piezo, *popt, k_piezo=K_PIEZO, lmbda=LAMBDA
                )
                plt.plot(
                    cas_ms,
                    u_teoria_fit,
                    "r--",
                    label=f"{subor.stem} - Fit teória ($R^2={r_squared:.3f}$)",
                    linewidth=1.8,
                )

        except Exception as e:
            print(f"Chyba pri spracovaní súboru {subor.name}: {e}")

    plt.title(
        "Odozva interferometra / polarizátora s automatickým fitom", fontsize=13
    )
    plt.xlabel("Čas [ms]", fontsize=11)
    plt.ylabel("Napätie [V]", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Cesta k priečinku alebo súboru
    MOJ_PRIECINOK = r"C:\Users\PC\Desktop\VS Code\V.semester\data\priem_signaly\scope_10.csv"

    #ch1 --> piezo, ch2 --> detektor

    nacitaj_a_vykresli_osciloskop_s_fitom(
        MOJ_PRIECINOK,
        show_ch1=True,
        scale_ch1=True,
        show_ch2=True,
        show_model=True,
    )