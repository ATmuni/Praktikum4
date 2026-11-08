import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('meranie_polarizacia.csv')
print(df.head(20))

plt.figure(figsize=(9, 5))
plt.plot(df['Uhol polarizatora'], df['Peak-peak napatie (uV)'], 'o-', color='#1f77b4', linewidth=2, markersize=6, label='Namerané dáta')

plt.title('Závislosť peak-peak napätia od uhla polarizátora', fontsize=14)
plt.xlabel('Uhol polarizátora [°]', fontsize=12)
plt.ylabel('Peak-peak napätie [μV]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

#plt.savefig('zavislost_napatia_od_uhla.png', dpi=300)
plt.show()
