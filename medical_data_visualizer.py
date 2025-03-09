import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = pd.Series( (df['weight'] / (df['height'] / 100) ** 2) > 25, dtype=int)

# 3
df['cholesterol'] = pd.Series(df['cholesterol'] > 1, dtype=int)
df['gluc'] = pd.Series(df['gluc'] > 1, dtype=int)
df.head(10)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # 6
    df_cat = pd.DataFrame(df_cat.groupby(['cardio', 'variable', 'value'])['value'].count()).rename(columns={'value': 'total'}).reset_index()
    

    # 7



    # 8
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').figure


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    pressure_mask = (df['ap_lo'] <= df['ap_hi'])
    low_height_mask = df['height'] >= df['height'].quantile(0.025)
    hi_height_mask = df['height'] <= df['height'].quantile(0.975)
    low_weight_mask = df['weight'] >= df['weight'].quantile(0.025)
    hi_weight_mask = df['weight'] <= df['weight'].quantile(0.975)
    
    df_heat = df[pressure_mask & low_height_mask & hi_height_mask & low_weight_mask & hi_weight_mask]

    # 12
    corr = df_heat.corr(numeric_only=True)

    # 13
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    # 14
    fig, ax = plt.subplots(figsize=(12, 12))


    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, cbar_kws={'shrink': 0.5})


    # 16
    fig.savefig('heatmap.png')
    return fig
