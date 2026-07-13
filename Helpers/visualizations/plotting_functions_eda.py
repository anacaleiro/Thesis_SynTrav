import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt


URBAN_SCORE = {
    'Very strongly urban': 5,
    'Strongly urban':      4,
    'Moderately urban':    3,
    'Not very urban':      2,
    'Not urban':           1,
}

NAME_MAP = {
    'Noord-Holland': 'North Holland',
    'Zuid-Holland':  'South-Holland',
    'Noord-Brabant': 'North Brabant',
    'Zeeland':       'Zealand',
    'Fryslân':       'Friesland',
}


def plot_province_urbanization(odin_cleaned, save_path=None):
    df = odin_cleaned.copy()
    df['urban_score'] = df['Urbanization class of residential municipality'].map(URBAN_SCORE)

    province_urban = (
        df.groupby('Province of residential municipality')['urban_score']
        .mean().reset_index()
        .rename(columns={'Province of residential municipality': 'province',
                         'urban_score': 'avg_urban_score'})
    )

    provinces_gdf = gpd.read_file(
        "https://cartomap.github.io/nl/wgs84/provincie_2022.geojson"
    )[['statnaam', 'geometry']].copy()
    provinces_gdf['province'] = provinces_gdf['statnaam'].map(NAME_MAP).fillna(provinces_gdf['statnaam'])

    merged = provinces_gdf.merge(province_urban, on='province', how='left').to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(7, 9))
    fig.patch.set_facecolor('white')
    merged.plot(
        column='avg_urban_score', ax=ax, cmap='YlOrRd', alpha=0.6, legend=True,
        legend_kwds={'orientation': 'vertical', 'shrink': 0.5, 'pad': 0.02, 'fraction': 0.04},
        edgecolor='white', linewidth=0.8, missing_kwds={'color': 'lightgrey'},
    )
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    for _, row in merged.iterrows():
        ax.text(row.geometry.centroid.x, row.geometry.centroid.y, row['province'],
                ha='center', va='center', fontsize=7.5, color='black', weight='bold')
    ax.set_axis_off()
    ax.set_title('Urbanization Level by Province', fontsize=13, pad=12, color='black')
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
