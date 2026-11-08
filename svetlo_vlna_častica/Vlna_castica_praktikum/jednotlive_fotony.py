from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ==========================================
# Fyzikálne parametre piezoaktuátora
# ==========================================
FREKVENCIA_PIEZO = 101.4  # Hz (z vášho vyhodnotenia piezoaktuátora)
PERIODA_PIEZO = 1.0 / FREKVENCIA_PIEZO  # s (~0.0098619 s = 9.8619 ms)

MOJ_PRIECINOK = r"C:\Users\PC\Desktop\VS Code\V.semester\data\jednotlive_fotony\scope_13.csv"


def deteguj_fotony_osciloskop(
    priecinok_path,
    u_min=0.5,
    distance=1,
    prominence=0.0001,
    f_piezo=FREKVENCIA_PIEZO,
):
    cesta = Path(priecinok_path)

    if not cesta.exists():
        print(f"Chyba: Cesta '{priecinok_path}' neexistuje.")
        return

    csv_subory = [cesta] if cesta.is_file() else sorted(list(cesta.glob("*.csv")))

    if not csv_subory:
        print("Nenašli sa žiadne CSV súbory.")
        return

    # Vypočítame periódu
    t_perioda = 1.0 / f_piezo

    for subor in csv_subory:
        try:
            df = pd.read_csv(
                subor,
                skiprows=2,
                header=None,
                names=["Time", "CH1", "CH2"],
                skipinitialspace=True,
            ).dropna()

            t = df["Time"].values
            ch1 = df["CH1"].values
            ch2 = df["CH2"].values

            # 1. Odčítanie strednej hladiny šumu na CH2
            ch2_baseline = np.median(ch2)
            ch2_centered = ch2 - ch2_baseline

            # 2. SPRÁVNY PREPOČET PERIÓD PIEZA
            celkovy_cas_zaznamu = np.ptp(t)  # t_max - t_min v sekundách
            pocet_period_v_zazname = celkovy_cas_zaznamu / t_perioda

            # 3. Detekcia pikov fotónov na CH2
            piky_idx, _ = find_peaks(
                ch2_centered,
                height=u_min,
                distance=distance,
                prominence=prominence,
            )

            celkovo_pikov = len(piky_idx)

            # Prepočet celkového počtu pikov presne na 1 periódu piezam
            pikov_na_1_periodu = celkovo_pikov / pocet_period_v_zazname

            print(f"==========================================")
            print(f" SÚBOR: {subor.name}")
            print(f" Trvanie záznamu osciloskopu: {celkovy_cas_zaznamu * 1e3:.4f} ms")
            print(f" Známa perióda piezoaktuátora: {t_perioda * 1e3:.4f} ms ({f_piezo} Hz)")
            print(f" Presný počet periód v tomto zázname: {pocet_period_v_zazname:.4f}")
            print(f" Celkový počet detegovaných pikov: {celkovo_pikov}")
            print(f" ---> POČET PIKOV NA 1 PERIÓDU: {pikov_na_1_periodu:.2f}")
            print(f"==========================================\n")

            # Vykreslenie grafu
            fig, ax1 = plt.subplots(figsize=(11, 6))

            ax1.plot(
                t * 1e3,
                ch2_centered,
                color="#1f77b4",
                linewidth=1.0,
                alpha=0.8,
                label="CH2 (Signál detektora)",
            )
            ax1.axhline(
                u_min,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"Prah $U_{{min}} = {u_min:.2f}$ mV",
            )

            if celkovo_pikov > 0:
                ax1.plot(
                    t[piky_idx] * 1e3,
                    ch2_centered[piky_idx],
                    "x",
                    color="red",
                    markersize=5,
                    markeredgewidth=1.2,
                    label=f"Detegované fotóny ({celkovo_pikov})",
                )

            # Sekundárna os pre budiaci sínus CH1
            ax2 = ax1.twinx()
            ax2.plot(
                t * 1e3,
                ch1,
                color="gray",
                linestyle=":",
                linewidth=1.0,
                alpha=0.4,
                label="CH1 ($U_{piezo}$)",
            )
            ax2.set_ylabel("Napätie CH1 [V]", color="gray", fontsize=11)

            ax1.set_title(
                f"Detekcia fotónov: {subor.name}\n({celkovo_pikov} pikov / {pocet_period_v_zazname:.3f} periódy = {pikov_na_1_periodu:.2f} pikov/periódu)",
                fontsize=13,
            )
            ax1.set_xlabel("Čas [ms]", fontsize=11)
            ax1.set_ylabel("Odozva CH2 [V]", fontsize=11)
            ax1.grid(True, linestyle="--", alpha=0.5)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Chyba pri spracovaní súboru {subor.name}: {e}")


if __name__ == "__main__":
    deteguj_fotony_osciloskop(
        MOJ_PRIECINOK,
        u_min=0.5,  
        distance=1,
        prominence=0.7,  
        f_piezo=101.4,  # Frekvencia piezoaktuátora z vášho merania (Hz)
    )