import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# NEWTONOVE KRUHY
#
# Fit:
#     r_k^2 = A + B k
#
# Podľa zadania:
#     B = (1/2) lambda R
#     lambda = 2B / R
#
# Hertzova škvrna:
#     rho = sqrt(A)       ... polomer Hertzovej škvrny
#     a = 2 sqrt(A)      ... priemer Hertzovej škvrny
# ============================================================


# Kalibrácia obrazu: 861 dielikov = 1 mm
DIELIKOV_NA_MM = 861.0


# Polomer zakrivenia šošovky v mm
R_MM = 82.3


# Dáta: (k, x_R, x_L), kde x sú v dielikoch obrazu.
data = {
    "RED": [
        (1, 174, 434),
        (2, 110, 491),
        (3, 70, 534),
        (4, 35, 572),
        (5, 0, 603),
    ],
    "ORANGE": [
        (1, 179, 427),
        (2, 117, 486),
        (3, 73, 527),
        (4, 36, 563),
        (5, 6, 596),
    ],
    "YELLOW": [
        (1, 198, 412),
        (2, 131, 468),
        (3, 84, 516),
        (4, 41, 550),
        (5, 12, 582),
    ],
    "YELLOW-GREEN": [
        (1, 191, 403),
        (2, 132, 466),
        (3, 87, 511),
        (4, 50, 549),
        (5, 21, 576),
    ],
    "GREEN 1": [
        (1, 197, 399),
        (2, 140, 458),
        (3, 96, 504),
        (4, 60, 538),
        (5, 29, 569),
    ],
    "GREEN 2": [
        (1, 194, 406),
        (2, 139, 461),
        (3, 96, 503),
        (4, 64, 540),
        (5, 28, 571),
    ],
    "BLUE": [
        (1, 206, 394),
        (2, 146, 445),
        (3, 101, 487),
        (4, 68, 521),
        (5, 37, 551),
    ],
    "RED_MERANIE 2": [
        (1, 192, 409),
        (2, 127, 481),
        (3, 77, 522),
        (4, 42, 561),
        (5, 8, 593),
    ],
    "ORANGE_MERANIE 2": [
        (1, 189, 408),
        (2, 127, 473),
        (3, 79, 518),
        (4, 45, 553),
        (5, 13, 586),
    ],
}


def prepare_data(rows):
    """Prepočíta polomery kružníc r_k a r_k^2 do mm a mm^2."""
    array = np.asarray(rows, dtype=float)

    k = array[:, 0]
    x_r = array[:, 1]
    x_l = array[:, 2]

    # D_k = |x_R - x_L|, r_k = D_k / 2
    r_mm = np.abs(x_r - x_l) / (2.0 * DIELIKOV_NA_MM)
    r2_mm2 = r_mm**2

    return k, r_mm, r2_mm2


def linear_fit(k, r2_mm2):
    """
    Lineárny fit:
        r_k^2 = A + B k

    Vráti A, B, ich neistoty a koeficient determinácie R^2.
    """
    coeff, covariance = np.polyfit(k, r2_mm2, deg=1, cov=True)

    B, A = coeff
    sigma_B, sigma_A = np.sqrt(np.diag(covariance))

    r2_fit = A + B * k

    ss_res = np.sum((r2_mm2 - r2_fit)**2)
    ss_tot = np.sum((r2_mm2 - np.mean(r2_mm2))**2)
    r_squared = 1.0 - ss_res / ss_tot

    return A, B, sigma_A, sigma_B, r_squared


def mean_and_standard_error(values):
    """
    Vráti aritmetický priemer a chybu aritmetického priemeru:
        u_priemeru = s / sqrt(N)
    """
    values = np.asarray(values, dtype=float)

    mean = np.mean(values)

    if len(values) > 1:
        std = np.std(values, ddof=1)
        sem = std / np.sqrt(len(values))
    else:
        std = 0.0
        sem = 0.0

    return mean, std, sem


# ============================================================
# VÝBER LED
# ============================================================

print("Dostupné dátové súbory:")
for name in data:
    print(f"- {name}")

led = input("\nZadaj LED/dátový súbor: ").strip().upper()

if led not in data:
    raise ValueError(
        f"Neplatný výber: {led}. "
        "Vyber jednu z možností vypísaných vyššie."
    )

if R_MM <= 0:
    raise ValueError("R_MM musí byť kladný polomer zakrivenia v mm.")


# ============================================================
# PRÍPRAVA DÁT A LINEÁRNY FIT
# ============================================================

k, r_mm, r2_mm2 = prepare_data(data[led])

A, B, sigma_A, sigma_B, r_squared = linear_fit(k, r2_mm2)


# ============================================================
# 1. VÝSLEDKY Z LINEÁRNEHO FITU
# ============================================================

# lambda_fit = 2B/R
lambda_fit_mm = 2.0 * B / R_MM
sigma_lambda_fit_mm = 2.0 * sigma_B / R_MM

lambda_fit_nm = lambda_fit_mm * 1e6
sigma_lambda_fit_nm = sigma_lambda_fit_mm * 1e6


# Hertzova škvrna z parametra A:
# rho = sqrt(A), a = 2sqrt(A)
if A > 0:
    rho_fit_mm = np.sqrt(A)
    sigma_rho_fit_mm = sigma_A / (2.0 * np.sqrt(A))

    a_fit_mm = 2.0 * np.sqrt(A)
    sigma_a_fit_mm = sigma_A / np.sqrt(A)

    rho_fit_um = rho_fit_mm * 1000.0
    sigma_rho_fit_um = sigma_rho_fit_mm * 1000.0

    a_fit_um = a_fit_mm * 1000.0
    sigma_a_fit_um = sigma_a_fit_mm * 1000.0

else:
    rho_fit_mm = np.nan
    sigma_rho_fit_mm = np.nan
    a_fit_mm = np.nan
    sigma_a_fit_mm = np.nan

    rho_fit_um = np.nan
    sigma_rho_fit_um = np.nan
    a_fit_um = np.nan
    sigma_a_fit_um = np.nan


# ============================================================
# 2. ARITMETICKÝ PRIEMER LAMBDA_k
#
# lambda_k = 2 r_k^2 / (R k)
# ============================================================

lambda_individual_mm = 2.0 * r2_mm2 / (R_MM * k)
lambda_individual_nm = lambda_individual_mm * 1e6

lambda_mean_nm, lambda_std_nm, lambda_sem_nm = mean_and_standard_error(
    lambda_individual_nm
)

lambda_mean_mm = lambda_mean_nm * 1e-6


# ============================================================
# 3. ARITMETICKÝ PRIEMER POLOMEROV HERTZOVEJ ŠKVRNY
#
# r_k^2 = rho^2 + (1/2) lambda R k
#
# rho_k = sqrt(r_k^2 - (1/2) lambda_mean R k)
# ============================================================

rho2_individual_mm2 = r2_mm2 - 0.5 * lambda_mean_mm * R_MM * k

# Kvôli chybe merania môžu niektoré hodnoty vyjsť mierne záporné.
# Také body nemožno použiť na reálny výpočet polomeru.
valid_rho = rho2_individual_mm2 > 0

rho_individual_mm = np.sqrt(rho2_individual_mm2[valid_rho])
rho_individual_um = rho_individual_mm * 1000.0

if len(rho_individual_um) > 0:
    rho_mean_um, rho_std_um, rho_sem_um = mean_and_standard_error(
        rho_individual_um
    )

    a_mean_um = 2.0 * rho_mean_um
    a_std_um = 2.0 * rho_std_um
    a_sem_um = 2.0 * rho_sem_um

else:
    rho_mean_um = np.nan
    rho_std_um = np.nan
    rho_sem_um = np.nan

    a_mean_um = np.nan
    a_std_um = np.nan
    a_sem_um = np.nan


# ============================================================
# VÝPIS VÝSLEDKOV
# ============================================================

print("\n" + "=" * 72)
print(f"LED / dáta: {led}")
print(f"Polomer zakrivenia: R = {R_MM:.4f} mm")
print("=" * 72)

print("\nTabuľka nameraných polomerov:")
print(
    f"{'k':>3} "
    f"{'r_k [mm]':>14} "
    f"{'r_k² [mm²]':>16} "
    f"{'lambda_k [nm]':>17}"
)

for k_i, r_i, r2_i, lambda_i in zip(
    k, r_mm, r2_mm2, lambda_individual_nm
):
    print(
        f"{int(k_i):>3} "
        f"{r_i:>14.7f} "
        f"{r2_i:>16.8f} "
        f"{lambda_i:>17.2f}"
    )


print("\n" + "-" * 72)
print("LINEÁRNY FIT")
print("r_k² = A + Bk")
print("-" * 72)

print(f"A = ({A:.8f} ± {sigma_A:.8f}) mm²")
print(f"B = ({B:.8f} ± {sigma_B:.8f}) mm²")
print(f"R² fitu = {r_squared:.6f}")

print("\nVlnová dĺžka z fitu:")
print(f"lambda_fit = ({lambda_fit_nm:.2f} ± {sigma_lambda_fit_nm:.2f}) nm")

if A > 0:
    print("\nHertzova škvrna z fitu:")
    print(
        f"Polomer rho_fit = "
        f"({rho_fit_um:.2f} ± {sigma_rho_fit_um:.2f}) µm"
    )
    print(
        f"Priemer a_fit = "
        f"({a_fit_um:.2f} ± {sigma_a_fit_um:.2f}) µm"
    )
else:
    print("\nHertzovu škvrnu z fitu nemožno určiť, pretože A <= 0.")


print("\n" + "-" * 72)
print("ARITMETICKÝ PRIEMER")
print("-" * 72)

print("\nVlnová dĺžka z jednotlivých kružníc:")
print(
    f"lambda_priemer = "
    f"({lambda_mean_nm:.2f} ± {lambda_sem_nm:.2f}) nm"
)
print(f"Smerodajná odchýlka lambda_k: {lambda_std_nm:.2f} nm")

if len(rho_individual_um) > 0:
    print("\nHertzova škvrna z aritmetického priemeru:")
    print(
        f"Polomer rho_priemer = "
        f"({rho_mean_um:.2f} ± {rho_sem_um:.2f}) µm"
    )
    print(
        f"Priemer a_priemer = "
        f"({a_mean_um:.2f} ± {a_sem_um:.2f}) µm"
    )
    print(
        f"Počet reálnych hodnôt rho_k použitých v priemere: "
        f"{len(rho_individual_um)} z {len(k)}"
    )
else:
    print(
        "\nZ aritmetického priemeru sa nepodarilo určiť reálny "
        "polomer Hertzovej škvrny."
    )


# ============================================================
# TABUĽKA PRE HERTZOVU ŠKVRNU Z PRIEMERU
# ============================================================

print("\nTabuľka pre výpočet Hertzovej škvrny z priemeru:")
print(
    f"{'k':>3} "
    f"{'rho_k² [mm²]':>18} "
    f"{'rho_k [µm]':>16}"
)

for i in range(len(k)):
    if rho2_individual_mm2[i] > 0:
        rho_i_um = np.sqrt(rho2_individual_mm2[i]) * 1000.0

        print(
            f"{int(k[i]):>3} "
            f"{rho2_individual_mm2[i]:>18.8f} "
            f"{rho_i_um:>16.2f}"
        )
    else:
        print(
            f"{int(k[i]):>3} "
            f"{rho2_individual_mm2[i]:>18.8f} "
            f"{'nereálne':>16}"
        )


# ============================================================
# GRAF NAMERANÝCH HODNÔT A FITOVANEJ PRIAMKY
# ============================================================

k_fit = np.linspace(0, np.max(k) + 0.25, 300)
r2_fit = A + B * k_fit

plt.figure(figsize=(9, 6))

plt.scatter(
    k,
    r2_mm2,
    color="tab:blue",
    linewidth=0.6,
    s=75,
    zorder=3,
    label="Namerané hodnoty"
)

plt.plot(
    k_fit,
    r2_fit,
    color="tab:red",
    linewidth=2.2,
    zorder=2,
    label=(
        "Lineárny fit\n"
        rf"$r_k^2 = ({A:.6f}) + ({B:.6f})k$ mm$^2$" "\n"
        rf"$R^2 = {r_squared:.5f}$"
    )
)

plt.xlabel("Poradie kružnice $k$", fontsize=12)
plt.ylabel(r"$r_k^2$ [mm$^2$]", fontsize=12)
plt.title(f"Závislosť $r_k^2$ od $k$ — {led}", fontsize=13)

plt.xticks(k)
plt.grid(True, linestyle="--", alpha=0.45)
plt.legend(fontsize=10)
plt.tight_layout()

#nazov_grafu = f"newtonove_kruhy_fit_{led.replace(' ', '_')}.png"

#plt.savefig(nazov_grafu, dpi=300, bbox_inches="tight")
plt.show()

#print(f"\nGraf bol uložený ako: {nazov_grafu}")