#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx

# ==========================
# 1) Daten laden & filtern
# ==========================
flights_path = "Train_with_Countries.csv"
airports_path = "airports.dat"

# Flüge
df_countries = pd.read_csv(flights_path)

# (Optional) Airports einlesen – hier nicht zwingend genutzt, aber falls du später erweiterst:
cols = ["airport_id", "name", "city", "country", "IATA", "ICAO",
        "latitude", "longitude", "altitude", "timezone", "dst",
        "tz_db", "type", "source"]
airports_df = pd.read_csv(airports_path, header=None, names=cols)
airports_df = airports_df.dropna(subset=["IATA"])
airports_df = airports_df[airports_df["IATA"] != "\\N"]

# Verspätungsfilter (wie bisher)
df_geo = df_countries[(df_countries["target"] > 15) &
                      (df_countries["target"] < 500)].copy()

# ==========================
# 2) Netzwerk (undirektional)
# ==========================
edges = df_geo[["DEPSTN", "ARRSTN"]].dropna().astype(str)
edges = edges[edges["DEPSTN"] != edges["ARRSTN"]]

G = nx.Graph()
G.add_edges_from(map(tuple, edges.values))

degrees = dict(G.degree())
nodes_list = list(G.nodes())

# Layout (stabil dank seed)
pos = nx.spring_layout(G, k=None, iterations=50, seed=42)

x_nodes = [pos[n][0] for n in nodes_list]
y_nodes = [pos[n][1] for n in nodes_list]
deg_vals = [degrees.get(n, 0) for n in nodes_list]

# ==========================
# 3) Kanten-Trace (ein Scatter mit None-Trennern)
# ==========================
x_edges, y_edges = [], []
for u, v in G.edges():
    x_edges += [pos[u][0], pos[v][0], None]
    y_edges += [pos[u][1], pos[v][1], None]

edge_trace = go.Scatter(
    x=x_edges, y=y_edges,
    mode="lines",
    # neutrale Linienfarbe (falls du auch Kanten farblich mappen willst, siehe Variante unten)
    line=dict(color="rgba(120,120,120,0.6)", width=0.7),
    hoverinfo="none",
    showlegend=False
)

# ==========================
# 4) Knoten-Trace (Viridis_r)
# ==========================
hover_text = [f"{n}<br>Connections: {degrees.get(n, 0)}" for n in nodes_list]

node_trace = go.Scatter(
    x=x_nodes,
    y=y_nodes,
    mode="markers",
    hoverinfo="text",
    text=hover_text,
    marker=dict(
        size=9,
        color=deg_vals,
        cmin=max(min(deg_vals) if deg_vals else 0, 0),
        cmax=max(deg_vals) if deg_vals else 1,
        colorscale="Viridis_r",  # <-- Viridis reversed
        line=dict(color="white", width=0.5),
        colorbar=dict(
            title=dict(text="Node Connections", font=dict(
                family="Lexend, Arial, sans-serif")),
            tickfont=dict(family="Lexend, Arial, sans-serif")
        )
    ),
    showlegend=False
)

# ==========================
# (Optional) Kanten farblich mappen mit Viridis_r
# Falls du Kanten auch farbig möchtest (z. B. nach Häufigkeit),
# kommentiere den obigen edge_trace aus und nutze diesen Block:
# ==========================
# from plotly.colors import sample_colorscale
# # Häufigkeit pro Kante (undirected)
# # Achtung: Wenn df_geo mehrere gleiche Routen hat, könntest du vorher zählen.
# edge_weights = {}
# for u, v in G.edges():
#     key = tuple(sorted((u, v)))
#     edge_weights[key] = edge_weights.get(key, 0) + 1
# w_vals = list(edge_weights.values())
# wmin, wmax = min(w_vals), max(w_vals)
# if wmax == wmin:
#     wmax = wmin + 1e-9
# x_edges, y_edges, edge_colors = [], [], []
# for u, v in G.edges():
#     x_edges += [pos[u][0], pos[v][0], None]
#     y_edges += [pos[u][1], pos[v][1], None]
#     w = edge_weights[tuple(sorted((u, v)))]
#     t = (w - wmin) / (wmax - wmin)
#     edge_colors.append(sample_colorscale("Viridis_r", t)[0])
# edge_trace = go.Scatter(
#     x=x_edges, y=y_edges,
#     mode="lines",
#     line=dict(color="rgba(120,120,120,0.2)", width=0.7),
#     hoverinfo="none",
#     showlegend=False
# )
# # Plotly kann pro Scatter nur eine Linienfarbe – bei mehreren Farben bräuchte man mehrere Traces.
# # Für einfache Umsetzung lassen wir die Kanten neutral und färben die Knoten (Standard oben).

# ==========================
# 5) Layout + Font
# ==========================
fig = go.Figure(data=[edge_trace, node_trace])

fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    hovermode="closest",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Lexend, Arial, sans-serif"),
    hoverlabel=dict(font=dict(family="Lexend, Arial, sans-serif"))
)

out_path = "network_predefined_turbo_3.html"
fig.write_html(out_path, include_plotlyjs="cdn")

# Google-Font + globales CSS injizieren (Lexend)
font_link = '<link href="https://fonts.googleapis.com/css2?family=Lexend&display=swap" rel="stylesheet">'
global_css = """
<style>
  html, body, .js-plotly-plot, .plotly, .plotly text, .colorbar, .hoverlayer,
  .g-gtitle, .g-xtitle, .g-ytitle, .infolayer text {
    font-family: 'Lexend', Arial, sans-serif !important;
  }
</style>
"""
with open(out_path, "r", encoding="utf-8") as f:
    html_content = f.read()
html_content = html_content.replace("<head>", f"<head>{font_link}{global_css}")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[OK] HTML gespeichert: {os.path.abspath(out_path)}")

# ==========================
# 6) Git-Helper & Push
# ==========================


def run(cmd: str, check: bool = True):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def in_git_repo() -> bool:
    try:
        subprocess.run("git rev-parse --is-inside-work-tree",
                       shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def rebase_in_progress() -> bool:
    git_dir = subprocess.run("git rev-parse --git-dir", shell=True,
                             capture_output=True, text=True)
    if git_dir.returncode != 0:
        return False
    g = git_dir.stdout.strip()
    # typische Rebase-Indikatoren
    return any(os.path.exists(os.path.join(g, p)) for p in ["rebase-merge", "rebase-apply"])


if in_git_repo():
    if rebase_in_progress():
        print("[Hinweis] Rebase läuft. Bitte Konflikte lösen und dann manuell `git rebase --continue` ausführen.")
    else:
        try:
            run(f"git add {out_path}")
            # Commit nur erstellen, wenn es Änderungen gibt
            diff = subprocess.run("git diff --cached --quiet", shell=True)
            if diff.returncode != 0:
                run(
                    f'git commit -m "Auto-update network chart HTML ({out_path})"')
            else:
                print("[Info] Keine Änderungen zum Committen.")
            # Remote abgleichen & pushen
            run("git fetch origin", check=False)
            # Versuche reguläres rebase-pull
            try:
                run("git pull --rebase origin main")
            except subprocess.CalledProcessError:
                # Fallback bei getrennten Historien
                run("git pull --rebase --allow-unrelated-histories origin main")
            run("git push origin main")
            print("[OK] Änderungen nach GitHub gepusht.")
        except subprocess.CalledProcessError as e:
            print(f"[Fehler] Git-Befehl fehlgeschlagen: {e}")
            print("Bitte prüfe Remote, Branch-Namen und ggf. Anmeldedaten (SSH/Token).")
else:
    print("[Hinweis] Kein Git-Repository erkannt. Führe `git init` aus und setze den Remote, z. B.:")
    print("  git init")
    print("  git branch -M main")
    print("  git remote add origin https://github.com/<USER>/<REPO>.git")
    print("  # dann Skript erneut laufen lassen")
