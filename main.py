"""NBA Player Scatter Plots.

Interfície Flet per triar equip i any; els gràfics de dispersió (amb
clústers K-Means) s'obren en un procés a part amb matplotlib.

Versió pensada per distribuir-se com a executable independent:
  * sense scikit-learn/scipy (K-Means propi en numpy): executable molt més
    petit i arrencada més ràpida
  * importacions pesades diferides (el procés de gràfics no carrega Flet)
  * errors a un registre en lloc de morir en silenci amb --noconsole
  * indicador de càrrega i avís d'error a la interfície

Empaquetat (vegeu les notes al final del fitxer).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Entorn (ha d'anar ABANS d'importar numpy / pandas / matplotlib)
# --------------------------------------------------------------------------- #
APP_NAME = 'NBAPlayers'

# Amb ~15 jugadors per equip, el multifil de BLAS/OpenMP només afegeix
# sobrecost i memòria.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')


def _app_data_dir() -> Path:
    """Carpeta d'usuari amb permís d'escriptura (registre, memòria cau de matplotlib)."""
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home()))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
    path = base / APP_NAME
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        path = Path(tempfile.gettempdir()) / APP_NAME
        path.mkdir(parents=True, exist_ok=True)
    return path


DATA_DIR = _app_data_dir()
LOG_FILE = DATA_DIR / 'error.log'

# La memòria cau de tipus de lletra de matplotlib es desa aquí: es construeix una sola vegada
os.environ.setdefault('MPLCONFIGDIR', str(DATA_DIR / 'matplotlib'))


def log_exception(context: str = '') -> None:
    """Desa el traceback actual al registre (mai llança excepcions)."""
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as fh:
            fh.write(f'\n--- {datetime.now():%Y-%m-%d %H:%M:%S} {context}\n')
            traceback.print_exc(file=fh)
    except OSError:
        pass


def _prepare_streams() -> None:
    """Amb --noconsole sys.stdout/sys.stderr són None i qualsevol print o
    avís d'una llibreria pot trencar l'aplicació. Es redirigeixen al registre."""
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 1_000_000:
            LOG_FILE.unlink()  # evita que el registre creixi sense límit
    except OSError:
        pass
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(LOG_FILE, 'a', encoding='utf-8')


_prepare_streams()

import argparse  # noqa: E402
import multiprocessing  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuració
# --------------------------------------------------------------------------- #
N_CLUSTERS = 5
COLORMAP = 'RdYlGn'  # vermell (baix) -> verd (alt)

# Mapatge de columnes a noms llegibles per a eixos i tooltips
AXIS_LABELS = {
    'fieldGoalsAttempted': 'Tirs de camp intentats',
    'fieldGoalsPercentage': 'Percentatge de tir de camp',
    'threePointersAttempted': 'Triples intentats',
    'threePointersMade': 'Triples anotats',
    'plusMinusPoints': 'Més/Menys',
    'numMinutes': 'Minuts jugats',
    'steals': 'Recuperacions',
    'blocks': 'Taps',
    'assists': 'Asistencies',
}


@dataclass(frozen=True)
class PlotSpec:
    """Defineix un scatter plot i com s'ordenen/anomenen els seus clústers.

    rank_by: què decideix que un clúster sigui "més alt" que un altre
        'x'   -> centroide a l'eix X
        'y'   -> centroide a l'eix Y
        'sum' -> suma dels dos eixos (estandarditzats)
    labels: noms de menor a major (han de ser N_CLUSTERS).
    """
    x: str
    y: str
    rank_by: str
    labels: tuple[str, ...]


PLOTS = [
    PlotSpec('fieldGoalsAttempted', 'fieldGoalsPercentage', 'y', (
        'Percentatge molt baix', 'Percentatge baix', 'Percentatge mig',
        'Percentatge alt', 'Percentatge molt alt')),
    PlotSpec('threePointersAttempted', 'threePointersMade', 'sum', (
        'Sense triple', 'Tirador ocasional', 'Tirador regular',
        'Tir bo', 'Gran tirador')),
    PlotSpec('plusMinusPoints', 'numMinutes', 'x', (
        'Impacte molt negativu', 'Impacte negatiu', 'Impacte neutre',
        'Impacte positiu', 'Impacte molt positiu')),
    PlotSpec('steals', 'blocks', 'sum', (
        'Defensa molt baixa', 'Defensa baixa', 'Defensa bona',
        'Defensa alta', 'Defensa d\'élit')),
    PlotSpec('steals', 'numMinutes', 'x', (
        'Moltes poques recuperacions', 'Poques recuperacions', 'Recuperacions mitjanes',
        'Moltes recuperacions', 'Recuperador d\'élit')),
    PlotSpec('assists', 'numMinutes', 'x', (
        'Moltes poques assistencies', 'Poques assistencies', 'Assistencies mitjanes',
        'Moltes assistencies', 'Creador de joc d\'élit')),
]


# --------------------------------------------------------------------------- #
# Utilitats
# --------------------------------------------------------------------------- #
def resource_path(relative: str) -> Path:
    """Ruta vàlida tant en desenvolupament com dins de l'executable (PyInstaller)."""
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
    return base / relative


def db():
    """Càrrega diferida de playersdb (pot llegir les dades en importar-se).

    Així un error de càrrega es pot mostrar a la interfície en lloc de matar
    l'aplicació abans que aparegui la finestra.
    """
    import playersdb
    return playersdb


def player_position(row) -> str:
    combo = (row['guard'], row['forward'], row['center'])
    return {
        (1, 1, 0): 'Escolta',
        (1, 0, 0): 'Base',
        (0, 1, 0): 'Aler',
        (0, 1, 1): 'Ala-Pívot',
        (0, 0, 1): 'Pívot',
    }.get(combo, 'Unknown')


def axis_label(column: str) -> str:
    return AXIS_LABELS.get(column, column)


def get_team_name(team_id: int) -> str:
    stats = db().playersStatisticsData
    names = stats.loc[stats['playerteamId'] == team_id, 'playerteamName'].dropna()
    return str(names.iloc[0]) if not names.empty else f'Team {team_id}'


# --------------------------------------------------------------------------- #
# Clustering (K-Means en numpy: evita empaquetar scikit-learn + scipy)
# --------------------------------------------------------------------------- #
def _assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return dist.argmin(axis=1)


def _kmeans_plus_plus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Inicialització k-means++ "greedy": a cada pas prova diversos candidats
    i es queda amb el que més redueix la distància total (com scikit-learn)."""
    n_trials = 2 + int(np.log(k))
    centers = np.empty((k, x.shape[1]))
    centers[0] = x[rng.integers(len(x))]
    dist = ((x - centers[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = dist.sum()
        if total > 0:
            candidates = rng.choice(len(x), size=n_trials, p=dist / total)
        else:
            candidates = rng.integers(len(x), size=n_trials)
        best_idx, best_dist, best_total = None, None, np.inf
        for idx in candidates:
            new_dist = np.minimum(dist, ((x - x[idx]) ** 2).sum(axis=1))
            if new_dist.sum() < best_total:
                best_idx, best_dist, best_total = idx, new_dist, new_dist.sum()
        centers[i] = x[best_idx]
        dist = best_dist
    return centers


def kmeans(x: np.ndarray, k: int, n_init: int = 20, max_iter: int = 300,
           tol: float = 1e-6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """K-Means amb inicialització k-means++ i diverses repeticions.

    Retorna (labels, centers) de la millor execució (menor inèrcia).
    """
    rng = np.random.default_rng(seed)
    best_inertia, best_labels, best_centers = np.inf, None, None

    for _ in range(n_init):
        centers = _kmeans_plus_plus(x, k, rng)
        for _ in range(max_iter):
            labels = _assign(x, centers)
            new_centers = centers.copy()
            for c in range(k):
                members = x[labels == c]
                if len(members):          # clúster buit: conserva el seu centre
                    new_centers[c] = members.mean(axis=0)
            shift = ((new_centers - centers) ** 2).sum()
            centers = new_centers
            if shift <= tol:
                break

        labels = _assign(x, centers)
        inertia = ((x - centers[labels]) ** 2).sum()
        if inertia < best_inertia - 1e-12:
            best_inertia, best_labels, best_centers = inertia, labels, centers

    return best_labels, best_centers


@dataclass
class ClusterResult:
    points: np.ndarray        # (n, 2) valors originals
    labels: np.ndarray        # cluster_id per jugador
    rank_of: np.ndarray       # rank_of[cluster_id] -> 0 (baix) .. k-1 (alt)
    centers: np.ndarray       # centroides en unitats originals
    names: list[str]          # nom per cluster_id
    colors: list[tuple]       # color per cluster_id


def cluster_players(data: pd.DataFrame, spec: PlotSpec) -> ClusterResult:
    import matplotlib  # diferit: accelera l'arrencada de la finestra Flet

    points = data[[spec.x, spec.y]].to_numpy(dtype=float)

    # Estandarditzat (equival a StandardScaler)
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    std[std == 0] = 1.0
    scaled = (points - mean) / std

    k = min(N_CLUSTERS, len(points))
    labels, centers_scaled = kmeans(scaled, k)

    # Posició de cada clúster -> ordre de menor a major
    if spec.rank_by == 'x':
        metric = centers_scaled[:, 0]
    elif spec.rank_by == 'y':
        metric = centers_scaled[:, 1]
    else:
        metric = centers_scaled.sum(axis=1)

    order = np.argsort(metric)
    rank_of = np.empty(k, dtype=int)
    rank_of[order] = np.arange(k)

    # Si hi ha menys de 5 clústers, reparteix els noms/colors al llarg de l'escala
    slots = np.linspace(0, len(spec.labels) - 1, k).round().astype(int)
    cmap = matplotlib.colormaps[COLORMAP]
    palette = cmap(np.linspace(0.08, 0.92, len(spec.labels)))

    names = [spec.labels[slots[rank_of[c]]] for c in range(k)]
    colors = [tuple(palette[slots[rank_of[c]]]) for c in range(k)]

    return ClusterResult(
        points=points,
        labels=labels,
        rank_of=rank_of,
        centers=centers_scaled * std + mean,
        names=names,
        colors=colors,
    )


# --------------------------------------------------------------------------- #
# Gràfics
# --------------------------------------------------------------------------- #
def style_axis(axis, spec: PlotSpec):
    axis.set_title(f'{axis_label(spec.x)} vs {axis_label(spec.y)}',
                   fontsize=11, fontweight='bold', loc='left')
    axis.set_xlabel(axis_label(spec.x), fontsize=9)
    axis.set_ylabel(axis_label(spec.y), fontsize=9)
    axis.grid(True, alpha=0.25, linestyle='--')
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    axis.tick_params(labelsize=8)


def draw_clusters(axis, result: ClusterResult):
    """Dibuixa els clústers de menor a major (així la llegenda surt ordenada)."""
    artists = []  # (artist, índexs de jugador)
    for cluster_id in np.argsort(result.rank_of):
        indices = np.flatnonzero(result.labels == cluster_id)
        if indices.size == 0:
            continue
        pts = result.points[indices]
        color = result.colors[cluster_id]
        name = result.names[cluster_id]

        artist = axis.scatter(
            pts[:, 0], pts[:, 1],
            s=55, color=color, alpha=0.85,
            edgecolors='white', linewidths=0.8,
            label=f'{name} ({indices.size})', zorder=3,
        )
        artists.append((artist, indices))

        # Marcador i nom al centroide del clúster
        cx, cy = result.centers[cluster_id]
        axis.annotate(
            name, xy=(cx, cy), xytext=(0, 12), textcoords='offset points',
            ha='center', fontsize=8, fontweight='bold', zorder=6,
            bbox={'boxstyle': 'round,pad=0.25', 'fc': 'white',
                  'ec': color, 'lw': 1.4, 'alpha': 0.92},
        )

    axis.legend(fontsize=7.5, loc='best', framealpha=0.9, title='Grups (nº jugadors)',
                title_fontsize=8)
    return artists


class HoverTooltip:
    """Tooltip amb dades del jugador en passar el ratolí per un punt."""

    def __init__(self, figure, data: pd.DataFrame):
        self.figure = figure
        self.data = data
        self.artists: dict = {}       # axis -> [(artist, indices)]
        self.annotations: dict = {}   # axis -> annotation
        self.current: dict = {}       # axis -> player_index | None

    def register(self, axis, artists):
        self.artists[axis] = artists
        annotation = axis.annotate(
            '', xy=(0, 0), xytext=(12, 12), textcoords='offset points',
            fontsize=8.5, zorder=20,
            bbox={'boxstyle': 'round,pad=0.5', 'fc': '#fffde7',
                  'ec': '#444', 'alpha': 0.97},
            arrowprops={'arrowstyle': '->', 'color': '#444'},
        )
        annotation.set_visible(False)
        self.annotations[axis] = annotation
        self.current[axis] = None

    def connect(self):
        self.figure.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.figure.canvas.mpl_connect('figure_leave_event', self._on_leave)

    def _find_player(self, event):
        # Només es comprova l'eix sota el cursor (més ràpid)
        for artist, indices in self.artists.get(event.inaxes, []):
            hit, details = artist.contains(event)
            if hit and len(details['ind']):
                i = details['ind'][0]
                return int(indices[i]), artist.get_offsets()[i]
        return None, None

    def _on_move(self, event):
        player_index, point = self._find_player(event)
        changed = False

        for axis, annotation in self.annotations.items():
            target = player_index if axis is event.inaxes else None
            if self.current[axis] == target:
                continue
            self.current[axis] = target
            changed = True
            annotation.set_visible(target is not None)
            if target is None:
                continue

            player = self.data.iloc[target]
            right = event.x > axis.bbox.x0 + axis.bbox.width / 2
            top = event.y > axis.bbox.y0 + axis.bbox.height / 2
            annotation.xy = point
            annotation.set_position((-14 if right else 14, -14 if top else 14))
            annotation.set_ha('right' if right else 'left')
            annotation.set_va('top' if top else 'bottom')
            annotation.set_text(
                f'{player["firstName_stats"]} {player["lastName_stats"]}\n'
                f'Altura: {player["heightInches"] * 2.54:.1f} cm\n'
                f'Pes: {player["bodyWeightLbs"] * 0.453592:.1f} kg\n'
                f'Posició: {player_position(player)}'
            )

        if changed:
            self.figure.canvas.draw_idle()

    def _on_leave(self, _event):
        for axis, annotation in self.annotations.items():
            self.current[axis] = None
            annotation.set_visible(False)
        self.figure.canvas.draw_idle()


class NoPlayersError(ValueError):
    """Equip/any sense jugadors: error 'esperat', s'avisa l'usuari."""


def load_team_data(team_id: int, year: int) -> pd.DataFrame:
    pdb = db()
    result = pdb.get_team_players(
        pdb.playersStatisticsData,
        pdb.playersData,
        selected_team_id=team_id,
        selected_year=year,
    )
    data = result[1] if isinstance(result, tuple) else result
    if data.empty:
        raise NoPlayersError('No s\'han trobat jugadors per a aquest equip i any.')
    return data.reset_index(drop=True)


def show_plots(team_id: int, year: int, ready_file: str | None = None):
    import matplotlib
    try:
        import tkinter  # noqa: F401  (use() no falla fins més tard si falta Tk)
        matplotlib.use('TkAgg')  # backend explícit: més fiable en empaquetar
    except Exception:
        pass  # sense Tk es deixa el backend per defecte de matplotlib
    import matplotlib.pyplot as plt

    data = load_team_data(team_id, year)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'figure.facecolor': '#f5f6fa',
        'axes.facecolor': 'white',
    })
    figure, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    figure.suptitle(
        f'{get_team_name(team_id)} · estadístiques de jugadors ({year})',
        fontsize=16, fontweight='bold',
    )
    try:
        figure.canvas.manager.set_window_title('Estadístiques de Jugadors NBA')
    except Exception:
        pass

    tooltip = HoverTooltip(figure, data)
    for axis, spec in zip(axes.flat, PLOTS):
        result = cluster_players(data, spec)
        artists = draw_clusters(axis, result)
        style_axis(axis, spec)
        tooltip.register(axis, artists)

    tooltip.connect()

    # Avisa la interfície que tot està calculat i que la finestra està a punt de sortir
    if ready_file:
        try:
            Path(ready_file).touch()
        except OSError:
            pass
    plt.show()


def run_plots_process(team_id: int, year: int, ready_file: str | None) -> int:
    """Punt d'entrada del procés de gràfics. Retorna el codi de sortida."""
    try:
        show_plots(team_id, year, ready_file)
        return 0
    except NoPlayersError as error:
        _show_message(str(error))
        return 0
    except Exception:
        log_exception(f'plots team={team_id} year={year}')
        return 1


def _show_message(text: str) -> None:
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showwarning('Estadístiques de Jugadors NBA', text)
        root.destroy()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Interfície Flet
# (Flet s'importa dins de cada funció: el procés de gràfics no el necessita)
# --------------------------------------------------------------------------- #
def create_team_dropdown():
    import flet as ft
    teams = (
        db().playersStatisticsData[['playerteamId', 'playerteamName']]
        .dropna()
        .drop_duplicates()
        .sort_values('playerteamName')
    )
    return ft.Dropdown(
        label='Equip',
        width=350,
        options=[
            ft.dropdown.Option(key=str(int(r.playerteamId)), text=r.playerteamName)
            for r in teams.itertuples()
        ],
        value=str(int(teams.iloc[0]['playerteamId'])),
    )


def create_year_dropdown():
    import flet as ft
    # Només es processen les dates úniques (milers) i no totes les files (milions)
    unique_dates = db().playersStatisticsData['gameDate'].drop_duplicates()
    dates = pd.to_datetime(unique_dates, errors='coerce')
    years = sorted(dates.dt.year.dropna().astype(int).unique())
    return ft.Dropdown(
        label='Any',
        width=350,
        options=[ft.dropdown.Option(str(y)) for y in years],
        value=str(years[-1]),
    )


def launch_plots_process(team_id: str, year: str, ready_file: Path) -> subprocess.Popen:
    """Llança els gràfics en un altre procés.

    En un executable (sys.frozen) no hi ha cap script per passar a l'intèrpret:
    el propi .exe es rellança amb --plots.
    """
    args = ['--plots', team_id, year, '--ready-file', str(ready_file)]
    if getattr(sys, 'frozen', False):
        command = [sys.executable, *args]
    else:
        command = [sys.executable, str(Path(__file__).resolve()), *args]

    env = os.environ.copy()
    # Si s'usa --onefile, el fill extreu la seva pròpia còpia en lloc de compartir
    # la carpeta temporal del pare (que s'esborra en tancar la finestra principal).
    env['PYINSTALLER_RESET_ENVIRONMENT'] = '1'

    kwargs = {}
    if sys.platform == 'win32':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

    # Amb --noconsole cal redirigir stdin/stdout/stderr explícitament;
    # si no, Popen pot fallar amb "handle no vàlid" a Windows.
    with open(LOG_FILE, 'a', encoding='utf-8') as log:
        return subprocess.Popen(
            command, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log,
            **kwargs,
        )


def main(page):
    import flet as ft

    page.title = 'Estadístiques de Jugadors NBA'
    page.window.width = 480
    page.window.height = 420
    page.padding = 30
    page.theme_mode = ft.ThemeMode.LIGHT

    status = ft.Text(size=13, color=ft.Colors.GREEN_700)

    try:
        team_dropdown = create_team_dropdown()
        year_dropdown = create_year_dropdown()
    except Exception:
        log_exception('carga de datos')
        page.add(
            ft.Text('No s\'han pogut carregar les dades.', size=18,
                    weight=ft.FontWeight.BOLD, color=ft.Colors.RED_700),
            ft.Text(f'Detalls a: {LOG_FILE}', size=12, selectable=True),
        )
        return

    spinner = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
    open_button = ft.Button('Obrir gràfics')

    def set_status(text: str, ok: bool = True, loading: bool = False):
        status.value = text
        status.color = ft.Colors.GREEN_700 if ok else ft.Colors.RED_700
        spinner.visible = loading
        open_button.disabled = loading
        try:
            page.update()
        except Exception:  # la finestra principal ja s'ha tancat
            pass

    def watch(process: subprocess.Popen, ready_file: Path):
        """Segueix el procés de gràfics i en reflecteix l'estat a la interfície."""
        ready = False
        while process.poll() is None:
            if not ready and ready_file.exists():
                ready = True
                set_status('Gràfics oberts en una finestra apart.')
            time.sleep(0.2)

        ready_file.unlink(missing_ok=True)
        if process.returncode != 0:
            set_status(f'Error en obrir els gràfics. Detalls a: {LOG_FILE}', ok=False)
        elif not ready:
            set_status('')  # avís ja mostrat pel procés (p. ex. sense jugadors)

    def start_plots(_):
        ready_file = DATA_DIR / f'ready_{uuid.uuid4().hex}'
        try:
            process = launch_plots_process(team_dropdown.value, year_dropdown.value,
                                           ready_file)
        except Exception as error:
            log_exception('lanzar proceso de gráficos')
            set_status(f'Error: {error}', ok=False)
            return
        set_status('Carregant gràfics… pot trigar uns segons.', loading=True)
        threading.Thread(target=watch, args=(process, ready_file), daemon=True).start()

    open_button.on_click = start_plots

    page.add(
        ft.Column(
            [
                ft.Text('🏀 Estadístiques de Jugadors NBA', size=24, weight=ft.FontWeight.BOLD),
                ft.Text('Selecciona equip i any', size=14, color=ft.Colors.GREY_700),
                team_dropdown,
                year_dropdown,
                ft.Row([open_button, spinner], spacing=12),
                status,
            ],
            spacing=18,
            tight=True,
        )
    )


# --------------------------------------------------------------------------- #
# Punt d'entrada
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots', nargs=2, type=int, metavar=('TEAM_ID', 'YEAR'))
    parser.add_argument('--ready-file')
    # parse_known_args: tolera arguments extra que afegeixi el runtime de Flet
    arguments, _unknown = parser.parse_known_args()
    return arguments


if __name__ == '__main__':
    multiprocessing.freeze_support()  # necessari en executables de Windows
    arguments = parse_args()

    if arguments.plots:
        sys.exit(run_plots_process(*arguments.plots, arguments.ready_file))

    try:
        import flet as ft
        ft.run(main)
    except Exception:
        log_exception('arranque')
        raise

# --------------------------------------------------------------------------- #
# Notes per crear l'executable (Windows; només s'ha provat el codi Python)
# --------------------------------------------------------------------------- #
# Recomanat: carpeta (--onedir) en lloc de --onefile. Amb --onefile, CADA
# arrencada descomprimeix tot a %TEMP% (diversos segons) i el procés de
# gràfics ho repeteix una segona vegada.
#
#   pip install flet pyinstaller numpy pandas matplotlib
#   flet pack main.py --name NBAPlayers --onedir --yes --add-data "datos:datos"
#
# Alternativa amb PyInstaller directe:
#   pyinstaller --noconsole --onedir --name NBAPlayers ^
#       --collect-all flet --add-data "datos;datos" ^
#       --exclude-module sklearn --exclude-module scipy ^
#       --exclude-module IPython --exclude-module pytest main.py
#
# Si playersdb llegeix CSV, carrega'ls amb resource_path('datos/fitxer.csv') perquè
# es trobin dins de l'executable i no relatius al directori actual.
#
# Per repartir-lo: comprimeix la carpeta dist/NBAPlayers en un .zip, o crea un
# instal·lador (p. ex. Inno Setup). Evita UPX: dispara falsos positius d'antivirus.
