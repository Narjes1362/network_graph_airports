import os
import subprocess
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# --------------------------
# 1. CSV-Daten einlesen
# --------------------------
flights_path = "Train_with_Countries.csv"
airports_path = "airports.dat"

df_countries = pd.read_csv(flights_path)
cols = ["airport_id", "name", "city", "country", "IATA", "ICAO", "latitude", "longitude",
        "altitude", "timezone", "dst", "tz_db", "type", "source"]
airports_df = pd.read_csv(airports_path, header=None, names=cols)
airports_df = airports_df.dropna(subset=["IATA", "latitude", "longitude"])
airports_df = airports_df[airports_df["IATA"] != "\\N"]

# --------------------------
# 2. Filter target > 15 und < 500
# --------------------------
df_geo = df_countries[(df_countries["target"] > 15) &
                      (df_countries["target"] < 500)].copy()

# --------------------------
# 3. Koordinaten anfügen
# --------------------------
iata_coords = airports_df[["IATA", "latitude",
                           "longitude"]].drop_duplicates("IATA")
df_geo = df_geo.merge(iata_coords, left_on="DEPSTN", right_on="IATA", how="left") \
               .rename(columns={"latitude": "dep_lat", "longitude": "dep_lon"}).drop(columns=["IATA"])
df_geo = df_geo.merge(iata_coords, left_on="ARRSTN", right_on="IATA", how="left") \
               .rename(columns={"latitude": "arr_lat", "longitude": "arr_lon"}).drop(columns=["IATA"])
df_geo = df_geo.dropna(subset=["dep_lat", "dep_lon", "arr_lat", "arr_lon"])

# --------------------------
# 4. Routen aggregieren (Edges)
# --------------------------
grouped_geo = df_geo.groupby(["DEPSTN", "ARRSTN", "dep_lat", "dep_lon", "arr_lat", "arr_lon"]).agg(
    flights=("target", "size"),
    mean_delay=("target", "mean")
).reset_index()

# Edge-Stats
edge_md = grouped_geo["mean_delay"].to_numpy()
ed_min, ed_max = float(np.nanmin(edge_md)), float(np.nanmax(edge_md))
if ed_max == ed_min:
    ed_max = ed_min + 1e-9
wmin, wmax = grouped_geo["flights"].min(), grouped_geo["flights"].max()


def w_to_width(w, wmin, wmax):
    if wmax == wmin:
        return 2.5
    return 0.5 + 5.5 * (w - wmin) / (wmax - wmin)


# --------------------------
# 5. Edge-Traces (Turbo)
# --------------------------
edge_traces = []
mid_lons, mid_lats, hover_texts = [], [], []

for _, r in grouped_geo.iterrows():
    lons = [r["dep_lon"], r["arr_lon"]]
    lats = [r["dep_lat"], r["arr_lat"]]
    t = (r["mean_delay"] - ed_min) / \
        (ed_max - ed_min) if ed_max > ed_min else 0.5
    color = sample_colorscale("Turbo", t)[0]
    edge_traces.append(go.Scattergeo(
        lon=lons, lat=lats,
        mode="lines",
        line=dict(width=w_to_width(r["flights"], wmin, wmax), color=color),
        hoverinfo="skip",
        showlegend=False
    ))
    mid_lons.append(np.mean(lons))
    mid_lats.append(np.mean(lats))
    hover_texts.append(
        f"Route: {r['DEPSTN']} → {r['ARRSTN']}<br>"
        f"Flights: {int(r['flights'])}<br>"
        f"Mean delay: {r['mean_delay']:.1f}"
    )

# Hoverpunkte für Edges
edge_hover = go.Scattergeo(
    lon=mid_lons, lat=mid_lats,
    mode="markers",
    marker=dict(size=6, opacity=0),
    hoverinfo="text",
    hovertext=hover_texts,
    showlegend=False
)

# Edge-Colorbar
edge_colorbar = go.Scattergeo(
    lon=[None], lat=[None],
    mode="markers",
    marker=dict(
        size=0.1,
        color=[ed_min, ed_max],
        colorscale="Turbo",
        showscale=True,
        colorbar=dict(
            title=dict(text="mean delay (Route)", font=dict(
                family="Lexend, Arial, sans-serif")),
            x=1.02,
            y=0.2,
            thickness=15,
            len=0.3,
            tickfont=dict(family="Lexend, Arial, sans-serif")
        )
    ),
    hoverinfo="none",
    showlegend=False
)

# --------------------------
# 6. Nodes (Viridis)
# --------------------------
dep_delay = df_geo.groupby("DEPSTN")["target"].mean().rename("mean_delay")
arr_delay = df_geo.groupby("ARRSTN")["target"].mean().rename("mean_delay")
node_md = pd.concat([dep_delay.rename_axis(
    "stn"), arr_delay.rename_axis("stn")]).groupby("stn").mean()

nodes_dep = grouped_geo[["DEPSTN", "dep_lat", "dep_lon"]].rename(
    columns={"DEPSTN": "stn", "dep_lat": "lat", "dep_lon": "lon"})
nodes_arr = grouped_geo[["ARRSTN", "arr_lat", "arr_lon"]].rename(
    columns={"ARRSTN": "stn", "arr_lat": "lat", "arr_lon": "lon"})
nodes = pd.concat([nodes_dep, nodes_arr],
                  ignore_index=True).drop_duplicates(subset=["stn"])
nodes = nodes.merge(node_md.to_frame().reset_index(), on="stn", how="left")

# Land pro IATA
dep_map = df_geo[["DEPSTN", "DEP_COUNTRY"]].rename(
    columns={"DEPSTN": "IATA", "DEP_COUNTRY": "country"})
arr_map = df_geo[["ARRSTN", "ARR_COUNTRY"]].rename(
    columns={"ARRSTN": "IATA", "ARR_COUNTRY": "country"})
iata_country = (
    pd.concat([dep_map, arr_map], ignore_index=True)
      .dropna(subset=["IATA", "country"])
      .groupby("IATA")["country"]
      .agg(lambda s: s.value_counts().idxmax())
      .reset_index()
)
nodes = nodes.merge(iata_country.rename(
    columns={"IATA": "stn"}), on="stn", how="left")
nodes["country"] = nodes["country"].fillna("Unknown")

nd_min, nd_max = float(np.nanmin(nodes["mean_delay"])), float(
    np.nanmax(nodes["mean_delay"]))
if nd_max == nd_min:
    nd_max = nd_min + 1e-9

node_trace = go.Scattergeo(
    lon=nodes["lon"], lat=nodes["lat"],
    mode="markers+text",
    text=nodes["stn"],
    textposition="top center",
    textfont=dict(family="Lexend, Arial, sans-serif"),
    hoverinfo="text",
    hovertext=[
        f"Airport: {s} ({c})<br>Node mean delay: {d:.1f}"
        for s, c, d in zip(nodes["stn"], nodes["country"], nodes["mean_delay"])
    ],
    marker=dict(
        size=10,
        color=nodes["mean_delay"],
        colorscale="Inferno_r",
        cmin=nd_min, cmax=nd_max,
        showscale=True,
        colorbar=dict(
            title=dict(text="mean delay (Airport)", font=dict(
                family="Lexend, Arial, sans-serif")),
            x=1.02,
            y=0.8,
            thickness=15,
            len=0.3,
            tickfont=dict(family="Lexend, Arial, sans-serif")
        ),
        line=dict(width=1, color="white")
    ),
    showlegend=False
)

# --------------------------
# 7. Figure erstellen
# --------------------------
fig = go.Figure(data=edge_traces + [edge_hover, edge_colorbar, node_trace])
# fig = go.Figure(data=node_trace)

fig.update_layout(
    title_x=0.5,
    geo=dict(
        projection_type="natural earth",
        showcountries=True,
        showland=True,
        showframe=True,
        framecolor="#5D7398",
        landcolor="#CCD7C7",
        countrycolor="#5D7398",
        coastlinecolor="#5D7398",
    ),
    margin=dict(l=10, r=10, t=60, b=10),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#5D7398",
    font=dict(
        family="Lexend, Arial, sans-serif",
        size=12,
        color="black"
    ),
    hoverlabel=dict(
        font=dict(family="Lexend, Arial, sans-serif")
    )
)

# --------------------------
# 8. Speichern
# --------------------------
out_path = "delay_airports_routes.html"
# out_path = "delay_airports_2.html"

fig.write_html(out_path, include_plotlyjs="cdn")


def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

# Falls das Repo lokal noch keinen Remote hat:
# run("git init")
# run("git remote add origin https://github.com/Narjes1362/network_graph_airports.git")


# Speichern der Datei
fig.write_html(out_path, include_plotlyjs="cdn")

# --- Lexend-Link + Global-CSS NACH dem letzten write_html injizieren ---
with open(out_path, "r", encoding="utf-8") as f:
    html_content = f.read()

font_link = '<link href="https://fonts.googleapis.com/css2?family=Lexend&display=swap" rel="stylesheet">'
global_css = """
<style>
  html, body, .js-plotly-plot, .plotly, .plotly text, .colorbar, .hoverlayer, .g-gtitle, .g-xtitle, .g-ytitle, .infolayer text {
    font-family: 'Lexend', Arial, sans-serif !important;
  }
</style>
"""
html_content = html_content.replace("<head>", f"<head>{font_link}{global_css}")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_content)
# -----------------------------------------------------------------------

# Git-Befehle mit Variable
# Git-Befehle mit Variable
run(f"git add {out_path}")
run(f'git commit -m "Auto-update chart HTML ({out_path})"')
run("git branch -M main")  # stellt sicher, dass der Branch 'main' heißt

# Force Push, um remote immer mit lokalem Stand zu überschreiben
run("git push origin main --force")

print(os.path.abspath(out_path))
