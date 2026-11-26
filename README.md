# 🏀 Estadístiques de Jugadors NBA

Aplicació d'escriptori per explorar les estadístiques dels jugadors d'un equip de l'NBA en una temporada concreta. Tria un **equip** i un **any**, i l'aplicació mostra sis gràfics de dispersió on els jugadors s'agrupen automàticament en **5 grups** (clusters K-Means) segons el seu rendiment.

<!-- Afegeix aquí una captura de pantalla de l'aplicació:
![Captura de l'aplicació](docs/captura.png)
-->

## Característiques

- Interfície senzilla feta amb [Flet](https://flet.dev): desplegables d'equip i any.
- Sis gràfics de dispersió en una sola finestra de [matplotlib](https://matplotlib.org):
  | Eix X | Eix Y |
  |---|---|
  | Tirs de camp intentats | Percentatge de tir de camp |
  | Triples intentats | Triples anotats |
  | Més/Menys | Minuts jugats |
  | Recuperacions | Taps |
  | Recuperacions | Minuts jugats |
  | Assistències | Minuts jugats |
- Agrupació dels jugadors amb K-Means (5 grups). Cada grup té un nom descriptiu (per exemple, *Gran tirador* o *Defensa d'élit*) i un color que va del vermell (baix) al verd (alt).
- **Tooltip interactiu**: en passar el ratolí per sobre d'un jugador es mostra el seu nom, l'alçada (cm), el pes (kg) i la posició.
- Pensada per funcionar com a programa independent: no cal tenir Python instal·lat.

## Descàrrega i ús (sense instal·lar Python)

1. Ves a l'apartat [**Releases**](../../releases) d'aquest repositori i descarrega l'últim `NBAPlayers.zip`.
2. Descomprimeix-lo.
3. Executa `NBAPlayers.exe`.
4. Tria l'equip i l'any, i prem **Obrir gràfics**. La finestra amb els gràfics pot trigar uns segons a aparèixer.

> **Nota:** Windows pot mostrar un avís de SmartScreen en executar-lo per primera vegada, perquè l'executable no està signat digitalment. Prem *Més informació → Executa igualment*.

## Executar des del codi font

### Requisits

- Python 3.10 o superior
- Les llibreries `flet`, `numpy`, `pandas` i `matplotlib`
- `tkinter` (ve inclòs amb Python a Windows i macOS; a Linux: `sudo apt install python3-tk`)

### Passos

```bash
git clone https://github.com/<el-teu-usuari>/<el-teu-repositori>.git
cd <el-teu-repositori>

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install flet numpy pandas matplotlib
python main.py
```

## Estructura del projecte

```
.
├── main.py          # Interfície Flet, clustering i gràfics
├── playersdb.py     # Càrrega i preparació de les dades dels jugadors
├── csv/             # Fitxers de dades
└── README.md
```

## Com funciona

1. **Interfície (Flet).** `main.py` mostra la finestra de selecció. En prémer el botó, llança els gràfics en un **procés separat** perquè la finestra principal no es bloquegi.
2. **Dades.** `playersdb.py` carrega les estadístiques i filtra els jugadors de l'equip i l'any triats.
3. **Clustering.** Per a cada gràfic, els valors dels dos eixos s'estandarditzen i s'agrupen amb un K-Means implementat amb NumPy (inicialització k-means++ i diverses repeticions). Els grups s'ordenen de menor a major segons l'eix o la combinació d'eixos definida per a aquell gràfic, i d'aquest ordre en surten el nom i el color.
4. **Gràfics.** Es dibuixen amb matplotlib (backend TkAgg) amb una llegenda per grup i el tooltip interactiu.

## Generar l'executable (Windows)

Necessites Python instal·lat. Posa `main.py`, `playersdb.py`, `build.bat` i la carpeta `datos` (si la fas servir) a la mateixa carpeta i executa:

```bat
build.bat
```

L'script crea un entorn virtual net, instal·la les dependències i genera l'executable amb `flet pack` en mode carpeta. El resultat és a `dist\NBAPlayers\NBAPlayers.exe`. Per repartir-lo, comprimeix tota la carpeta `dist\NBAPlayers` en un `.zip`.

Si prefereixes fer-ho a mà:

```bat
pip install flet pyinstaller numpy pandas matplotlib
flet pack main.py --name NBAPlayers --onedir --yes --add-data "datos:datos"
```

### Per què carpeta (`--onedir`) i no un sol `.exe`?

Amb `--onefile`, cada arrencada descomprimeix tot el contingut a una carpeta temporal, cosa que fa que l'aplicació trigui diversos segons a obrir-se. Com que els gràfics s'obren en un segon procés, aquest cost es repeteix. El mode carpeta arrenca molt més ràpid.

## Resolució de problemes

Si alguna cosa falla, l'aplicació ho indica a la finestra i desa els detalls tècnics en un fitxer de registre:

| Sistema | Ruta del registre |
|---|---|
| Windows | `%LOCALAPPDATA%\NBAPlayers\error.log` |
| macOS | `~/Library/Application Support/NBAPlayers/error.log` |
| Linux | `~/.local/share/NBAPlayers/error.log` |

Si obres una *issue*, adjunta el contingut d'aquest fitxer.

## Dades

<!-- Indica aquí l'origen de les dades (per exemple, el conjunt de dades de Kaggle o l'API que fas servir), la seva llicència i l'última data d'actualització. -->

## Llicència

<!-- Tria una llicència (per exemple, MIT) i afegeix el fitxer LICENSE al repositori. -->
