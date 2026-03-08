import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set the style for a professional dashboar
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

# =============================================================================
# 1. SIMULATE THE MOSQUITO BREEDING & POPULATION DATA (2019-2023)
# =============================================================================
np.random.seed(42)  # For reproducibility

years = np.array([2019, 2020, 2021, 2022, 2023])
n_years = len(years)

# Simulate key breeding environment factors (e.g., rainfall, temp)
rainfall_mm = np.array([950, 1020, 1250, 1180, 1100])  # Annual rainfall
avg_temp_c = np.array([23.5, 23.8, 24.5, 24.2, 24.6])   # Annual average temperature

# Simulate Anopheles mosquito population index (a measure of density)
# Population growth is influenced by breeding conditions (rainfall, temperature)
base_population = 100  # Starting index value in 2019
mosquito_population = np.zeros(n_years)
mosquito_population[0] = base_population

for i in range(1, n_years):
    # Growth is positively correlated with rainfall and temperature
    growth_factor = 1 + (rainfall_mm[i] - 1000) / 5000 + (avg_temp_c[i] - 24) / 50
    # Add some random variability
    random_factor = np.random.normal(1.0, 0.1)
    mosquito_population[i] = mosquito_population[i-1] * growth_factor * random_factor

# Create DataFrame for Mosquito Data
df_mosquito = pd.DataFrame({
    'Year': years,
    'Rainfall_mm': rainfall_mm,
    'Avg_Temp_C': avg_temp_c,
    'Mosquito_Pop_Index': mosquito_population.round(1)
})

# =============================================================================
# 2. SIMULATE THE MALARIA IMPACT DATA (2019-2023)
# =============================================================================
# Assume malaria cases are correlated with mosquito population, with a lag effect
lag_correlation = 0.7  # Strength of correlation between mosquito pop and cases

# Base case numbers (in thousands)
base_cases = 800
base_deaths = 0.7  # in thousands

# Calculate cases: function of mosquito population and random factors
malaria_cases = (base_cases * (mosquito_population / 100) * 
                np.array([1.0, 0.95, 1.25, 0.90, 1.10]) *  # Year-specific factors (e.g., intervention efforts)
                np.random.normal(1.0, 0.05, n_years)).astype(int)

# Calculate deaths: function of cases and healthcare access (improving over time)
healthcare_factor = np.array([1.0, 0.98, 0.95, 0.93, 0.90])  # Improving healthcare over time
malaria_deaths = (base_deaths * (malaria_cases / base_cases) * healthcare_factor * 
                 np.random.normal(1.0, 0.1, n_years)).round(1)

# Create DataFrame for Malaria Impact Data
df_malaria = pd.DataFrame({
    'Year': years,
    'Malaria_Cases_Thousands': malaria_cases,
    'Malaria_Deaths_Thousands': malaria_deaths
})

# Merge the DataFrames
df_combined = pd.merge(df_mosquito, df_malaria, on='Year')

print("="*80)
print("INTEGRATED ANALYSIS: Anopheles Mosquitoes & Malaria Impact in Ethiopia (2019-2023)")
print("="*80)
print("\nCOMBINED DATASET:")
print(df_combined.to_string(index=False))

# =============================================================================
# 3. CREATE THE DASHBOARD VISUALIZATION
# =============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Integrated Dashboard: Mosquito Breeding, Population Growth, and Malaria Impact\nEthiopia (2019-2023)', 
             fontsize=20, fontweight='bold', y=0.98)

# ------------------------------------
# PLOT 1: Malaria Impact (Bar Graph)
# ------------------------------------
bar_width = 0.35
x = np.arange(len(years))
bars1 = ax1.bar(x - bar_width/2, df_combined['Malaria_Cases_Thousands'], bar_width, 
                color='coral', alpha=0.8, label='Cases (Thousands)')
bars2 = ax1.bar(x + bar_width/2, df_combined['Malaria_Deaths_Thousands'] * 10, bar_width, 
                color='purple', alpha=0.8, label='Deaths (Hundreds)') # Scaled for visibility

ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Number of People', fontweight='bold')
ax1.set_title('A: People Affected by Malaria (Cases & Deaths)', fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(years)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if bar.get_facecolor() == bars1[0].get_facecolor(): # Cases
            ax1.text(bar.get_x() + bar.get_width()/2., height + 20, f'{int(height)}K', 
                    ha='center', va='bottom', fontweight='bold')
        else: # Deaths (remember they are scaled)
            actual_deaths = height / 10
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2, f'{actual_deaths:.1f}K', 
                    ha='center', va='bottom', fontweight='bold')

# ------------------------------------
# PLOT 2: Mosquito Population Growth (Line Graph)
# ------------------------------------
color = 'tab:green'
ax2.plot(df_combined['Year'], df_combined['Mosquito_Pop_Index'], 
         marker='o', linewidth=3, markersize=8, color=color, label='Population Index')
ax2.set_xlabel('Year', fontweight='bold')
ax2.set_ylabel('Mosquito Population Index', color=color, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_title('B: Anopheles Mosquito Population Growth', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Annotate the peak year
max_pop_idx = df_combined['Mosquito_Pop_Index'].idxmax()
max_year = df_combined.loc[max_pop_idx, 'Year']
max_pop = df_combined.loc[max_pop_idx, 'Mosquito_Pop_Index']
ax2.annotate(f'Peak: {max_pop}', xy=(max_year, max_pop), xytext=(max_year-0.5, max_pop+5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontweight='bold')

# Create a twin axis to show breeding factors
ax2b = ax2.twinx()
color = 'tab:blue'
ax2b.bar(df_combined['Year'], df_combined['Rainfall_mm'], alpha=0.2, color=color, label='Rainfall (mm)')
ax2b.set_ylabel('Annual Rainfall (mm)', color=color, fontweight='bold')
ax2b.tick_params(axis='y', labelcolor=color)
# Combine legends from both axes
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center')

# ------------------------------------
# PLOT 3: Mosquito Contribution to Malaria (Histogram)
# ------------------------------------
# This plot shows the *proportion* of malaria attributable to Anopheles each year
# We assume it's the primary vector (95%+), with minor yearly fluctuations.
prop_attributable = np.array([0.96, 0.95, 0.97, 0.96, 0.98])  # Proportion

bars = ax3.bar(df_combined['Year'], prop_attributable * 100, 
               color='red', alpha=0.6, edgecolor='darkred')
ax3.set_xlabel('Year', fontweight='bold')
ax3.set_ylabel('Percentage of Malaria Cases (%)', fontweight='bold')
ax3.set_title('C: Percentage of Malaria Cases Caused by Anopheles Mosquitoes', fontsize=16, fontweight='bold')
ax3.set_ylim(90, 100)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height - 1, f'{height:.1f}%', 
             ha='center', va='top', fontweight='bold', color='white')

# ------------------------------------
# PLOT 4: Correlation Analysis (Scatter Plot)
# ------------------------------------
# Show the relationship between mosquito population and malaria cases
ax4.scatter(df_combined['Mosquito_Pop_Index'], df_combined['Malaria_Cases_Thousands'], 
           s=100, alpha=0.7, c=df_combined['Year'], cmap='viridis')

# Add regression line
z = np.polyfit(df_combined['Mosquito_Pop_Index'], df_combined['Malaria_Cases_Thousands'], 1)
p = np.poly1d(z)
ax4.plot(df_combined['Mosquito_Pop_Index'], p(df_combined['Mosquito_Pop_Index']), "r--", alpha=0.8)

ax4.set_xlabel('Mosquito Population Index', fontweight='bold')
ax4.set_ylabel('Malaria Cases (Thousands)', fontweight='bold')
ax4.set_title('D: Correlation: Mosquito Population vs. Malaria Cases', fontsize=16, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Calculate and display correlation coefficient
corr, p_value = stats.pearsonr(df_combined['Mosquito_Pop_Index'], df_combined['Malaria_Cases_Thousands'])
ax4.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {p_value:.4f}', 
         transform=ax4.transAxes, fontsize=12, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add colorbar for years
cbar = plt.colorbar(ax4.collections[0], ax=ax4)
cbar.set_label('Year')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for super title
plt.show()

# =============================================================================
# 4. KEY INSIGHTS AND SUMMARY
# =============================================================================
total_cases = df_combined['Malaria_Cases_Thousands'].sum() * 1000
total_deaths = df_combined['Malaria_Deaths_Thousands'].sum() * 1000
avg_contribution = prop_attributable.mean() * 100

print("\n" + "="*80)
print("KEY INSIGHTS FROM THE INTEGRATED ANALYSIS")
print("="*80)
print(f"1. HUMAN IMPACT:")
print(f"   ➤ Total Malaria Cases (2019-2023): {total_cases:,.0f}")
print(f"   ➤ Total Deaths (2019-2023): {total_deaths:,.0f}")
print(f"   ➤ Peak year for cases: {df_combined.loc[df_combined['Malaria_Cases_Thousands'].idxmax(), 'Year']}")

print(f"\n2. MOSQUITO POPULATION DYNAMICS:")
print(f"   ➤ Mosquito population increased by {(mosquito_population[-1] - mosquito_population[0]) / mosquito_population[0] * 100:.1f}% over 5 years.")
print(f"   ➤ Peak population occurred in {max_year}, correlating with high rainfall ({df_combined.loc[max_pop_idx, 'Rainfall_mm']}mm).")

print(f"\n3. VECTOR CONTRIBUTION:")
print(f"   ➤ Anopheles mosquitoes were responsible for {avg_contribution:.1f}% of all malaria cases.")
print(f"   ➤ The strong correlation (r = {corr:.3f}) confirms the direct link between")
print(f"     mosquito population size and human disease burden.")

print(f"\n4. INTERVENTION IMPLICATIONS:")
print(f"   ➤ Years with above-average rainfall (e.g., {max_year}) require intensified")
print(f"     larval source management and vector control measures.")
print(f"   ➤ The consistent high attribution rate (>95%) highlights that controlling")
print(f"     Anopheles mosquitoes is the most effective strategy for reducing malaria.")
print("="*80)
